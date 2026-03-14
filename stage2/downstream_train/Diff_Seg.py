import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from skimage import io
import torchvision.transforms as transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
import random
import time
from PIL import Image
import cv2
from losses import SSLoss, FLoss
from model import SegNet

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 超参数（5类别病理数据集适配）
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_CHANNELS = 3
BATCH_SIZE = 8  # 根据实际显存调整
LEARNING_RATE = 1e-4
EPOCHS = 150  # 5类分割可适当增加训练轮次
NUM_CLASSES = 5  # 5类别分割任务（包含背景）
# 注意：添加背景约束后，模型输出会临时增加1个通道用于约束
TRAIN_IMAGE_DIR = './img_renamed'  # 图像文件夹路径
TRAIN_MASK_DIR = './mask_renamed'  # mask文件夹路径
MODEL_TYPE = 'luad_diff_SSFL_withmaskpretrain_1.19'  # 带背景约束的5类别模型标识
LOSS_TYPE = 'SSFL_' + str(BATCH_SIZE)

# 5个类别的颜色映射（BGR格式，因为OpenCV默认读取为BGR）
COLOR_MAP = {
    (51, 51, 205): 0,  # TE (原RGB: (205,51,51) 转换为BGR)
    (0, 255, 0): 1,  # NEC (原RGB: (0,255,0))
    (225, 105, 65): 2,  # LYM (原RGB: (65,105,225) 转换为BGR)
    (0, 165, 255): 3,  # TAS (原RGB: (255,165,0) 转换为BGR)
    (255, 255, 255): 4  # bg (背景，原RGB: (255,255,255))
}

# 数据转换
transformations = transforms.Compose([
    transforms.ToTensor(),
])


def color_to_class(mask):
    """将彩色mask转换为类别索引"""
    # 转换为numpy数组并确保是uint8类型
    mask_np = np.uint8(mask * 255) if mask.max() <= 1 else np.uint8(mask)

    # 如果是3通道，转换为BGR格式便于与COLOR_MAP匹配
    if mask_np.ndim == 3 and mask_np.shape[2] == 3:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2BGR)

    h, w = mask_np.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)  # 存储类别索引

    # 遍历颜色映射，将对应颜色的像素赋值为相应类别
    for color, cls in COLOR_MAP.items():
        if mask_np.ndim == 3:
            # 3通道图像匹配
            match = np.all(mask_np == color, axis=-1)
        else:
            # 单通道图像匹配（如果有）
            match = (mask_np == color[0])
        class_mask[match] = cls

    return class_mask


def my_transforms(image1, mask):
    """数据增强：随机翻转、高斯模糊、颜色抖动"""
    if random.random() > 0.5:
        image1 = TF.vflip(image1)
        mask = TF.vflip(mask)

    if random.random() > 0.5:
        image1 = TF.hflip(image1)
        mask = TF.hflip(mask)

    if random.random() > 0.7:
        image1 = TF.gaussian_blur(image1, [3, 3], [1.0, 2.0])

    if random.random() > 0.7:
        jitter = transforms.ColorJitter(brightness=.5, contrast=.4)
        image1 = jitter(image1)

    return image1, mask


class EarlyStopping:
    """早停机制：当验证损失不再下降时停止训练"""

    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if epoch is not None:
            weight_path = f"{self.path[:-4]}_{epoch}_{val_loss:.5f}.pth"
        else:
            weight_path = self.path
        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'model_state_dict': model.state_dict(),
        }, weight_path)
        self.val_loss_min = val_loss


def get_images_and_masks_list(img_dir, mask_dir, k=None):
    """获取图像和mask文件名列表（确保一一对应）"""
    # 获取所有图像和mask文件
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 排序文件（按数字前缀）
    img_files.sort(key=lambda x: int(x.split('-')[0]))
    mask_files.sort(key=lambda x: int(x.split('-')[0]))

    # 确保图像和mask数量匹配
    assert len(img_files) == len(mask_files), "图像和mask数量不匹配"

    # 确保文件名一一对应
    for img_f, mask_f in zip(img_files, mask_files):
        assert img_f == mask_f, f"文件名不匹配: {img_f} 和 {mask_f}"

    if k is not None:
        return np.array(img_files[:k]), np.array(mask_files[:k])
    else:
        return np.array(img_files), np.array(mask_files)


class Histo_Dataset(Dataset):
    """5类别病理数据集加载器"""

    def __init__(self, image_dir, mask_dir, image_list, mask_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # 加载图像
        img_path = os.path.join(self.image_dir, self.image_list[index])
        image = io.imread(img_path)
        if image.ndim == 2:  # 如果是灰度图，转换为RGB
            image = np.stack([image] * 3, axis=-1)

        # 加载mask并转换为类别索引
        mask_path = os.path.join(self.mask_dir, self.mask_list[index])
        mask = io.imread(mask_path)
        mask = color_to_class(mask)  # 转换为类别索引

        if self.transform is not None:
            image = self.transform(image)
            mask = torch.from_numpy(mask).unsqueeze(0).float()  # 增加通道维度

        # 应用数据增强
        image, mask = my_transforms(image, mask)

        # 转换mask为长整数类型
        mask = mask.squeeze(0).long()

        return image, mask


def initialize_weights(model):
    """初始化模型权重"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight.data, 0.0, 0.01)


def F1_score(y_true, y_pred):
    """计算每个类别的F1分数"""
    class_f1_scores = []
    _, y_true = torch.max(y_true, 0)
    _, y_pred = torch.max(y_pred, 0)
    for i in range(NUM_CLASSES):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        class_f1_scores.append(f1_score(true, pred, zero_division=1))
    return class_f1_scores


def class_weights(y_predict, y_true):
    """计算类别权重（基于F1分数）"""
    fscores = np.zeros(NUM_CLASSES)
    for i in range(y_true.shape[0]):
        fscores += F1_score(y_true[i], y_predict[i])
    return fscores


def apply_background_constraint(output, target):
    """
    应用背景区域强制约束
    将输出与背景掩码拼接，强制背景区域（target == 4）预测为背景
    """
    # 创建背景约束掩码：背景区域权重设为2，其他区域为0
    batch_size, _, height, width = output.shape
    bg_mask = torch.ones((batch_size, 1, height, width), device=DEVICE) * 2.0
    bg_mask = bg_mask * (target == 4).unsqueeze(1).float()  # 只在背景区域应用约束

    # 将约束掩码拼接到输出通道维度上
    output_with_constraint = torch.cat([output, bg_mask], dim=1)
    return output_with_constraint


def train_epoch(train_loader, model, optimizer, epoch):
    """训练一轮"""
    model.train()
    losses = []
    p_bar = tqdm(train_loader)

    for h, true_label in p_bar:
        h = h.to(DEVICE)
        true_label = true_label.to(DEVICE)  # 5类别标签

        # 转换为one-hot编码（保持原类别数）
        target_label = F.one_hot(true_label, NUM_CLASSES)
        target_label = target_label.permute(0, 3, 1, 2).float()

        # 扩散模型时间步固定为0
        t = torch.full((h.shape[0],), 0, dtype=torch.long, device=DEVICE)
        predicted_label = model(h, t)  # 原始输出：[batch, NUM_CLASSES, H, W]

        # 应用背景约束：拼接背景掩码，输出变为[batch, NUM_CLASSES+1, H, W]
        # predicted_with_constraint = apply_background_constraint(predicted_label, true_label)
        # 在train_epoch中添加
        predicted_with_constraint = apply_background_constraint(predicted_label, true_label)
        # print("约束后输出的最大值：", predicted_with_constraint.max().item())
        # print("约束后输出的最小值：", predicted_with_constraint.min().item())
        # 计算损失时，需要调整目标以匹配新的输出维度
        # 将目标中背景区域(4)标记为新的类别索引NUM_CLASSES
        constrained_target = true_label.clone()
        constrained_target[true_label == 4] = NUM_CLASSES
        # 打印标签范围，确保无异常值
        # print("原始标签范围：", true_label.min().item(), true_label.max().item())  # 应在0~4
        # print("约束后标签范围：", constrained_target.min().item(), constrained_target.max().item())  # 应在0~5
        constrained_target_onehot = F.one_hot(constrained_target, NUM_CLASSES + 1)
        constrained_target_onehot = constrained_target_onehot.permute(0, 3, 1, 2).float()

        # 计算损失（SSLoss + FLoss）
        ss_loss = SSLoss()
        f_loss = FLoss(2.0)
        loss = ss_loss(predicted_with_constraint, constrained_target_onehot) + \
               f_loss(predicted_with_constraint, constrained_target_onehot)

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()

        # 在train_epoch的loss.backward()后添加
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 限制梯度范数

        optimizer.step()

        p_bar.set_description(f'Epoch {epoch}')
        p_bar.set_postfix(loss=loss.item())

    avg_loss = np.mean(losses)
    print(f'Epoch: {epoch}\tTrain Loss: {avg_loss:.4f}')
    return avg_loss


def eval_epoch(eval_loader, model, epoch):
    """验证一轮"""
    model.eval()
    val_loss = []
    p_bar = tqdm(eval_loader)
    total_items = 0.0
    f_scores = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for h, true_label in p_bar:
            h = h.to(DEVICE)
            true_label = true_label.to(DEVICE)

            # 转换为one-hot编码
            target_label = F.one_hot(true_label, NUM_CLASSES)
            target_label = target_label.permute(0, 3, 1, 2).float()

            t = torch.full((h.shape[0],), 0, dtype=torch.long, device=DEVICE)
            predicted_label = model(h, t)  # 原始输出

            # 应用背景约束
            predicted_with_constraint = apply_background_constraint(predicted_label, true_label)

            # 调整目标以匹配约束后的输出
            constrained_target = true_label.clone()
            constrained_target[true_label == 4] = NUM_CLASSES
            constrained_target_onehot = F.one_hot(constrained_target, NUM_CLASSES + 1)
            constrained_target_onehot = constrained_target_onehot.permute(0, 3, 1, 2).float()

            # 计算验证损失
            ss_loss = SSLoss()
            f_loss = FLoss(2.0)
            loss = ss_loss(predicted_with_constraint, constrained_target_onehot) + \
                   f_loss(predicted_with_constraint, constrained_target_onehot)
            val_loss.append(loss.item())

            # 计算F1分数时使用原始输出和原始目标（不包含约束通道）
            # 确保预测结果中背景区域被正确标记
            pred_argmax = torch.argmax(predicted_label, dim=1)
            pred_argmax[true_label == 4] = 4  # 强制背景区域预测为背景

            # 转换为one-hot用于F1计算
            pred_onehot = F.one_hot(pred_argmax, NUM_CLASSES)
            pred_onehot = pred_onehot.permute(0, 3, 1, 2).float()

            f_scores += class_weights(pred_onehot, target_label)
            total_items += target_label.shape[0]

            p_bar.set_description(f'Epoch {epoch}')
            p_bar.set_postfix(loss=loss.item())

    # 计算平均F1
    f_scores /= total_items
    print(f'Class F1 Scores: {f_scores}')
    print(f'Mean F1 Score: {np.mean(f_scores):.4f}')

    avg_val_loss = np.mean(val_loss)
    print(f'Epoch: {epoch}\tVal Loss: {avg_val_loss:.4f}')
    return avg_val_loss


def main():
    # 确保导入cv2（用于颜色空间转换）
    global cv2
    import cv2

    # 创建输出目录
    checkpoint_path = f'./snapshots/{MODEL_TYPE}/'
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(f'./plots/{MODEL_TYPE}', exist_ok=True)

    # 加载图像和mask文件列表
    img_list, mask_list = get_images_and_masks_list(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
    print(f"找到 {len(img_list)} 对图像和mask")

    # 划分训练集和验证集（9:1）
    ratio = 0.9
    idxs = np.random.RandomState(2023).permutation(len(img_list))
    split = int(len(img_list) * ratio)
    train_index, valid_index = idxs[:split], idxs[split:]

    train_dataset = Histo_Dataset(
        TRAIN_IMAGE_DIR, TRAIN_MASK_DIR,
        img_list[train_index], mask_list[train_index],
        transform=transformations
    )
    eval_dataset = Histo_Dataset(
        TRAIN_IMAGE_DIR, TRAIN_MASK_DIR,
        img_list[valid_index], mask_list[valid_index],
        transform=transformations
    )

    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=False, num_workers=4, pin_memory=True
    )

    # 初始化模型
    model = SegNet(dim=64, channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    initialize_weights(model)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 加载预训练权重（如果有）
    pretrained_ckpt = './8_0.0001_38_0.04583.pth'  # 替换为实际路径
    if os.path.exists(pretrained_ckpt):
        print(f'从 {pretrained_ckpt} 加载预训练权重')
        checkpoint = torch.load(pretrained_ckpt, map_location=DEVICE, weights_only=False)
        # 过滤掉分割头的权重（仅加载编码器/解码器部分）
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'final_conv' not in k}
        # ========== 新增：处理通道不匹配问题 ==========
        if 'net.init_conv.weight' in pretrained_dict:
            # 预训练权重的init_conv.weight形状是 [out_dim, 4, 7, 7]
            # 当前模型是 [out_dim, 3, 7, 7]，只保留前3个输入通道的权重
            init_conv_weight = pretrained_dict['net.init_conv.weight']
            # 取前3个输入通道（维度1）的权重，保持输出通道、核尺寸不变
            pretrained_dict['net.init_conv.weight'] = init_conv_weight[:, :3, :, :]
        # ===========================================

        model.load_state_dict(pretrained_dict, strict=False)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每10epoch衰减一半
    # 早停机制
    early_stopping = EarlyStopping(
        patience=20, verbose=True, path=f'{checkpoint_path}{BATCH_SIZE}_{LEARNING_RATE}.pth'
    )

    # 训练循环
    train_losses, eval_losses = [], []
    start_time = time.process_time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(train_dataloader, model, optimizer, epoch)
        eval_loss = eval_epoch(eval_dataloader, model, epoch)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        scheduler.step()  # 学习率衰减
        # 早停检查
        early_stopping(eval_loss, model, epoch)
        if early_stopping.early_stop:
            print('触发早停机制！')
            break

    # 保存损失曲线
    print(f"总训练时间: {time.process_time() - start_time:.2f} 秒")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
    plt.plot(range(1, len(eval_losses) + 1), eval_losses, 'r-', label='验证损失')
    plt.title(f'5类别病理分割训练曲线 (带背景约束，损失: SS + Focal)', fontsize=10)
    plt.xlabel('轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.savefig(f'./plots/{MODEL_TYPE}/loss_curve.jpg', dpi=300)


if __name__ == '__main__':
    main()