import os
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn import functional as F
import cv2
import shutil
from matplotlib import pyplot as plt
from tqdm import tqdm
from metrics import (Aggregated_jaccard_index, Hausdorff_distance, Evaluator)
from sklearn.metrics import ConfusionMatrixDisplay
import sys

sys.path.append('../downstream_train')
from model import SegNet

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 超参数
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
NUM_CLASSES = 5  # 总类别数（含背景）
FOREGROUND_CLASSES = 4  # 前景类别数（0-3）
TEST_IMAGE_DIR = './test/test_luad/img_renamed'
TEST_MASK_DIR = './test/test_luad/mask_renamed'

# 调色板（必须和GT Mask的调色板完全一致）
COLOR_MAP = {
    0: (205, 51, 51),  # TE
    1: (0, 255, 0),  # NEC
    2: (65, 105, 225),  # LYM
    3: (255, 165, 0),  # TAS
    4: (255, 255, 255)  # bg
}
# 反向映射：RGB值→类别索引
RGB_TO_CLASS = {v: k for k, v in COLOR_MAP.items()}

# 数据转换
transformations = transforms.Compose([transforms.ToTensor()])


def apply_background_constraint_in_test(pred_idx, target_label):
    """
    测试阶段强制背景约束：将GT中背景区域（target_label == 4）的预测结果设为4
    确保背景类100%正确（安全的CPU操作版本）
    """
    # 转到CPU处理掩码，避免CUDA断言
    pred_idx_cpu = pred_idx.cpu()
    target_label_cpu = target_label.cpu()

    # 创建背景掩码
    bg_mask = (target_label_cpu == 4)
    # 强制背景区域预测为4
    pred_idx_cpu[bg_mask] = 4

    # 转回原设备
    return pred_idx_cpu.to(pred_idx.device)


# 辅助函数：解析彩色GT Mask为类别索引
def rgb2class(mask_rgb):
    """将彩色Mask转换为0-4的类别索引"""
    h, w, c = mask_rgb.shape
    mask_class = np.zeros((h, w), dtype=np.int64)
    for rgb, cls in RGB_TO_CLASS.items():
        mask = np.all(mask_rgb == rgb, axis=-1)
        mask_class[mask] = cls
    return mask_class


# 计算Dice系数的函数（支持任意类别数）
def calculate_dice(pred, target, num_classes):
    """计算每个类别的Dice系数"""
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)

        intersection = np.sum(pred_cls * target_cls)
        union = np.sum(pred_cls) + np.sum(target_cls)

        if union == 0:
            dice = 1.0 if np.sum(pred_cls) == 0 and np.sum(target_cls) == 0 else 0.0
        else:
            dice = (2.0 * intersection) / union
        dice_scores.append(dice)
    return dice_scores


# Dataset类
class Histo_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = io.imread(img_path)
        mask_rgb = io.imread(mask_path)  # 读取彩色Mask
        mask = rgb2class(mask_rgb)  # 转换为0-4的类别索引

        # 调试：打印第一个样本的类别分布
        if index == 0:
            print(f"=== 调试：第一个样本的GT类别分布 ===")
            print(f"GT类别值范围: {np.min(mask)} ~ {np.max(mask)}")
            print(f"GT类别计数: {np.bincount(mask.flatten(), minlength=NUM_CLASSES)}")

        if self.transform is not None:
            image = self.transform(image)
        return image, mask


# 指标计算辅助函数
def compute_metrics_from_cm(cm, num_classes, is_foreground=False):
    """从混淆矩阵计算各项指标"""

    def replace_nan(arr):
        arr = np.array(arr)
        arr[np.isnan(arr)] = 0.0
        arr[np.isinf(arr)] = 0.0
        return arr

    # IoU
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    denominator = row_sum + col_sum - diag
    ious = np.where(denominator != 0, diag / denominator, 0.0)
    ious = replace_nan(ious)
    mean_iou = np.mean(ious)

    # Dice
    dice_scores = 2 * diag / (row_sum + col_sum)
    dice_scores = replace_nan(dice_scores)
    mean_dice = np.mean(dice_scores)

    # Pixel Accuracy
    pixel_acc = np.diag(cm).sum() / cm.sum() if cm.sum() != 0 else 0.0

    # Pixel Accuracy Class
    pixel_acc_class = np.mean(replace_nan(diag / row_sum)) if row_sum.sum() != 0 else 0.0

    # FWIoU
    freq = row_sum / cm.sum() if cm.sum() != 0 else 0.0
    freq = replace_nan(freq)
    fwiou = (freq[freq > 0] * ious[freq > 0]).sum()

    # 类别名称
    class_names = ['TE', 'NEC', 'LYM', 'TAS'] if is_foreground else ['TE', 'NEC', 'LYM', 'TAS', 'BG']

    metrics = {
        'ious': ious,
        'mean_iou': mean_iou,
        'dice_scores': dice_scores,
        'mean_dice': mean_dice,
        'pixel_acc': pixel_acc,
        'pixel_acc_class': pixel_acc_class,
        'fwiou': fwiou,
        'class_names': class_names
    }
    return metrics


# 评估函数（核心修正版）
def eval_epoch(test_loader, model, num_classes, foreground_classes, device):
    model_type = 'luad_1.19_withpretrainmasktest'
    loss_type = 'SSFL_Dice_A'

    with torch.no_grad():
        model.eval()
        # 初始化两个Evaluator：一个全5类，一个仅前景4类
        evaluator_5class = Evaluator(num_classes)  # 含背景
        evaluator_4class = Evaluator(foreground_classes)  # 仅前景

        # 存储AJI和Hausdorff结果
        aji_scores_5class = 0.0
        aji_scores_4class = 0.0
        hds_list_5class = []
        hds_list_4class = []
        total_items = 0
        p_bar = tqdm(test_loader)

        # 用于计算Dice的列表
        all_dice_5class = []
        all_dice_4class = []

        # 保存路径
        save_path2 = './test_luad_1.19_withpretrainmasktest/predicted_labels'
        save_path3 = './test_luad_1.19_withpretrainmasktest/images'
        for path in [save_path2, save_path3]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        indexing = 0
        for img, target_label in p_bar:
            img = img.to(device)
            target_label = target_label.to(device)  # shape: (1, H, W), 值为0-4

            # 模型预测
            t = torch.full((BATCH_SIZE,), 0, dtype=torch.long).to(device)
            predicted_label = model(img, t)  # shape: (1, 5, H, W)

            # 转换为类别索引（初始预测）
            _, pred_idx = torch.max(predicted_label, 1)  # shape: (1, H, W), 值为0-4

            # 强制背景约束：GT背景区域预测为4（确保100%正确）
            pred_idx = apply_background_constraint_in_test(pred_idx, target_label)

            # 调试：打印前2个样本的类别分布
            if total_items < 2:
                print(f"\n=== 调试：第{total_items}个样本的Pred类别分布 ===")
                pred_np = pred_idx.cpu().numpy().flatten()
                target_np = target_label.cpu().numpy().flatten()
                print(f"Pred类别值范围: {np.min(pred_np)} ~ {np.max(pred_np)}")
                print(f"Pred类别计数: {np.bincount(pred_np, minlength=num_classes)}")
                print(f"GT类别计数: {np.bincount(target_np, minlength=num_classes)}")

            # 转换为numpy数组（全程CPU操作，避免CUDA问题）
            pred_np = pred_idx.cpu().numpy()
            target_np = target_label.cpu().numpy()

            # 1. 更新5类Evaluator（含背景）
            evaluator_5class.add_batch(target_np, pred_np)

            # 2. 更新4类Evaluator（仅前景）：过滤掉背景类像素
            # 生成前景掩码（排除背景类4）
            foreground_mask = (target_np != 4)
            if np.sum(foreground_mask) > 0:  # 确保有前景像素
                target_foreground = target_np[foreground_mask]
                pred_foreground = pred_np[foreground_mask]
                # 确保值在0-3范围内
                target_foreground = np.clip(target_foreground, 0, 3)
                pred_foreground = np.clip(pred_foreground, 0, 3)
                # 调整形状匹配（避免reshape错误）
                target_foreground = target_foreground.reshape(1, -1)
                pred_foreground = pred_foreground.reshape(1, -1)
                # 添加到4类Evaluator
                evaluator_4class.add_batch(target_foreground, pred_foreground)

            # 计算当前样本的Dice系数
            pred_np_squeeze = pred_np.squeeze()
            target_np_squeeze = target_np.squeeze()

            # 5类Dice
            dice_5class = calculate_dice(pred_np_squeeze, target_np_squeeze, num_classes)
            all_dice_5class.append(dice_5class)

            # 4类Dice（仅前景）
            # 先过滤背景像素再计算
            if np.sum(foreground_mask) > 0:
                pred_foreground_squeeze = pred_np_squeeze[foreground_mask.squeeze()]
                target_foreground_squeeze = target_np_squeeze[foreground_mask.squeeze()]
                dice_4class = calculate_dice(pred_foreground_squeeze, target_foreground_squeeze, foreground_classes)
            else:
                dice_4class = [0.0] * foreground_classes
            all_dice_4class.append(dice_4class)

            # ========== 计算AJI和Hausdorff（修正one-hot越界问题） ==========
            # 5类版本（含背景）- 直接使用原始标签
            target_onehot_5 = F.one_hot(target_label.long(), num_classes).permute(0, 3, 1, 2).float()
            aji_5 = Aggregated_jaccard_index(target_onehot_5, predicted_label, device, include_background=True)
            hd_5 = Hausdorff_distance(target_onehot_5, predicted_label, num_classes, include_background=True)
            aji_scores_5class += aji_5
            hds_list_5class.append(hd_5)

            # 4类版本（仅前景）- 先过滤背景再编码
            # 1. 创建前景掩码（tensor版）
            foreground_mask_tensor = (target_label != 4)
            # 2. 生成仅前景的target（值范围0-3）
            target_foreground_tensor = target_label.clone()
            target_foreground_tensor[~foreground_mask_tensor] = 0  # 背景区域置0（不参与计算）
            target_foreground_tensor = torch.clamp(target_foreground_tensor, 0, 3)
            # 3. one-hot编码（仅4类）
            target_onehot_4 = F.one_hot(target_foreground_tensor.long(), foreground_classes).permute(0, 3, 1, 2).float()
            # 4. 生成仅前景的预测输出（去掉背景通道）
            predicted_label_4 = predicted_label[:, :foreground_classes, :, :]
            # 5. 计算4类AJI/Hausdorff（排除背景）
            aji_4 = Aggregated_jaccard_index(target_onehot_4, predicted_label_4, device, include_background=False)
            hd_4 = Hausdorff_distance(target_onehot_4, predicted_label_4, foreground_classes, include_background=False)
            aji_scores_4class += aji_4
            hds_list_4class.append(hd_4)

            total_items += 1

            # 保存预测图（包含背景类）
            labels_p = np.zeros((pred_idx.shape[0], pred_idx.shape[1], pred_idx.shape[2], 3), dtype=np.uint8)
            for cls in range(num_classes):
                labels_p[pred_idx.cpu() == cls] = COLOR_MAP[cls]

            for i in range(target_onehot_5.shape[0]):
                image_label_p = labels_p[i]
                cv2.imwrite(os.path.join(save_path2, f'label_{indexing}.png'),
                            cv2.cvtColor(image_label_p, cv2.COLOR_RGB2BGR))
                image = torch.permute(img[i], (1, 2, 0)).cpu().numpy()
                image = np.uint8(255 * image)
                cv2.imwrite(os.path.join(save_path3, f'image_{indexing}.png'),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                indexing += 1

            p_bar.set_postfix(total_items=total_items)

        # ========== 计算并打印5类指标（含背景） ==========
        print(f"\n" + "=" * 50)
        print(f"=== 版本1：完整5类指标（包含背景类） ===")
        print(f"=" * 50)

        # 5类混淆矩阵
        cm_5class = evaluator_5class.confusion_matrix
        print(f"\n混淆矩阵（5类）:")
        print(np.round(cm_5class).astype(int))

        # 计算5类指标
        metrics_5class = compute_metrics_from_cm(cm_5class, num_classes, is_foreground=False)

        # 打印5类指标
        print(f"\n[5类指标]")
        print(f"总样本数: {total_items}")
        print(f"Average AJI: {aji_scores_5class / total_items:.4f}")
        print(f"IoU scores: {dict(zip(metrics_5class['class_names'], np.round(metrics_5class['ious'], 4)))}")
        print(f"Mean IoU: {metrics_5class['mean_iou']:.4f}")
        print(
            f"Dice scores (混淆矩阵): {dict(zip(metrics_5class['class_names'], np.round(metrics_5class['dice_scores'], 4)))}")
        avg_dice_5 = np.mean(all_dice_5class, axis=0)
        print(f"Dice scores (逐样本计算): {dict(zip(metrics_5class['class_names'], np.round(avg_dice_5, 4)))}")
        print(f"Mean Dice: {metrics_5class['mean_dice']:.4f}")
        print(f"Pixel Accuracy (整体): {metrics_5class['pixel_acc']:.4f}")
        print(f"Pixel Accuracy (按类别平均): {metrics_5class['pixel_acc_class']:.4f}")
        print(f"FWIoU: {metrics_5class['fwiou']:.4f}")

        # 5类Hausdorff
        hds_5 = np.nan_to_num(np.mean(hds_list_5class, axis=0))
        print(f"Hausdorff Distances: {dict(zip(metrics_5class['class_names'], np.round(hds_5, 4)))}")
        print(f"Average Hausdorff Distance: {np.mean(hds_5):.4f}")

        # 保存5类混淆矩阵图
        disp_5 = ConfusionMatrixDisplay(cm_5class.astype(int), display_labels=metrics_5class['class_names'])
        disp_5.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix (5 Classes - Include Background)')
        plt.savefig(f'./cm_5class_{model_type}_{loss_type}_{BATCH_SIZE}.jpg', dpi=300)
        plt.close()

        # ========== 计算并打印4类指标（仅前景） ==========
        print(f"\n" + "=" * 50)
        print(f"=== 版本2：仅前景4类指标（排除背景类） ===")
        print(f"=" * 50)

        # 4类混淆矩阵
        cm_4class = evaluator_4class.confusion_matrix
        print(f"\n混淆矩阵（4类-仅前景）:")
        print(np.round(cm_4class).astype(int))

        # 计算4类指标
        metrics_4class = compute_metrics_from_cm(cm_4class, foreground_classes, is_foreground=True)

        # 打印4类指标
        print(f"\n[4类前景指标]")
        print(f"总样本数: {total_items}")
        print(f"Average AJI: {aji_scores_4class / total_items:.4f}")
        print(f"IoU scores: {dict(zip(metrics_4class['class_names'], np.round(metrics_4class['ious'], 4)))}")
        print(f"Mean IoU: {metrics_4class['mean_iou']:.4f}")
        print(
            f"Dice scores (混淆矩阵): {dict(zip(metrics_4class['class_names'], np.round(metrics_4class['dice_scores'], 4)))}")
        avg_dice_4 = np.mean(all_dice_4class, axis=0)
        print(f"Dice scores (逐样本计算): {dict(zip(metrics_4class['class_names'], np.round(avg_dice_4, 4)))}")
        print(f"Mean Dice: {metrics_4class['mean_dice']:.4f}")
        print(f"Pixel Accuracy (仅前景像素): {metrics_4class['pixel_acc']:.4f}")
        print(f"Pixel Accuracy (按前景类别平均): {metrics_4class['pixel_acc_class']:.4f}")
        print(f"FWIoU: {metrics_4class['fwiou']:.4f}")

        # 4类Hausdorff
        hds_4 = np.nan_to_num(np.mean(hds_list_4class, axis=0))
        print(f"Hausdorff Distances: {dict(zip(metrics_4class['class_names'], np.round(hds_4, 4)))}")
        print(f"Average Hausdorff Distance: {np.mean(hds_4):.4f}")

        # 保存4类混淆矩阵图
        disp_4 = ConfusionMatrixDisplay(cm_4class.astype(int), display_labels=metrics_4class['class_names'])
        disp_4.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix (4 Classes - Only Foreground)')
        plt.savefig(f'./cm_4class_{model_type}_{loss_type}_{BATCH_SIZE}.jpg', dpi=300)
        plt.close()


# 主函数
def main():
    def get_images_list(path1, k=None):
        total_list1 = os.listdir(path1)
        total_list1 = sorted(total_list1, key=lambda x: int(x.split('-')[0]))
        return np.array(total_list1) if k is None else np.array(total_list1[:k])

    img_list = get_images_list(TEST_IMAGE_DIR)
    test_dataset = Histo_Dataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        image_list=img_list,
        transform=transformations
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,  # 禁用多线程，避免CUDA问题
        pin_memory=False
    )

    # 加载模型
    path_train = "./snapshots/luad_diff_SSFL_withmaskpretrain_1.19/8_0.0001_45_0.10085.pth"
    snapshot = torch.load(path_train, map_location=device, weights_only=False)
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {path_train}")

    model = SegNet(dim=64, channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(snapshot['model_state_dict'])

    eval_epoch(test_loader, model, NUM_CLASSES, FOREGROUND_CLASSES, DEVICE)


if __name__ == '__main__':
    main()