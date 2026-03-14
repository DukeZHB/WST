import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from skimage import io
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
import re  # 新增：解析图像名称中的类别标签

IMG_SIZE = 256

# 新增：颜色映射（BGR转类别索引）
COLOR_MAP = {
    (51, 51, 205): 0,  # TE (原RGB: (205,51,51) 转换为BGR)
    (0, 255, 0): 1,  # NEC (原RGB: (0,255,0))
    (225, 105, 65): 2,  # LYM (原RGB: (65,105,225) 转换为BGR)
    (0, 165, 255): 3,  # TAS (原RGB: (255,165,0) 转换为BGR)
    (255, 255, 255): 4  # bg (背景，原RGB: (255,255,255))
}


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, betas_schedule, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(betas_schedule['sqrt_alphas_cumprod'], t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        betas_schedule['sqrt_one_minus_alphas_cumprod'], t, x_0.shape
    )
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def _my_normalization(x):
    return (x * 2) - 1


def get_images_list(path1, k=None):
    total_list1 = os.listdir(path1)
    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    total_list1 = [f for f in total_list1 if f.lower().endswith(supported_formats)]
    total_list1 = sorted(total_list1, key=lambda x: int(x.split('-')[0]))
    if k is None:
        return np.array(total_list1)
    else:
        return np.array(total_list1[:k])


# 新增：解析图像名称中的类别标签
def parse_category_label(filename):
    pattern = r'\[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]'
    match = re.search(pattern, filename)
    if match:
        labels = [float(match.group(i)) for i in range(1, 5)]
        return torch.FloatTensor(labels)
    else:
        return torch.FloatTensor([0., 0., 0., 0.])


# 新增：将伪掩码转换为类别索引图
def mask_to_class_index(mask_img):
    h, w, _ = mask_img.shape
    class_idx = np.zeros((h, w), dtype=np.int64)
    for color, idx in COLOR_MAP.items():
        mask = np.all(mask_img == color, axis=-1)
        class_idx[mask] = idx
    return class_idx


# 新增：掩码的变换（和原图保持一致）
def get_mask_transforms():
    mask_transforms = [
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]
    return transforms.Compose(mask_transforms)


class Histo_Dataset(Dataset):
    def __init__(self, image1_dir, mask_dir, image1_list, transform=None, mask_transform=None):
        self.image1_dir = image1_dir
        self.mask_dir = mask_dir
        self.image1_list = image1_list
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        # 加载原图
        img1_name = self.image1_list[index]
        img1_path = os.path.join(self.image1_dir, img1_name)
        image1 = io.imread(img1_path)

        # 加载伪掩码
        mask_path = os.path.join(self.mask_dir, img1_name)
        mask_img = io.imread(mask_path)

        # 解析类别标签
        cat_label = parse_category_label(img1_name)

        # 原图变换
        if self.transform is not None:
            image1 = self.transform(image1)

        # 掩码变换 + 转类别索引
        if self.mask_transform is not None:
            mask_img = self.mask_transform(mask_img)
            mask_img_np = mask_img.permute(1, 2, 0).numpy().astype(np.uint8)
            mask_class = mask_to_class_index(mask_img_np)
            mask_class = torch.LongTensor(mask_class)
        else:
            mask_class = torch.LongTensor(mask_to_class_index(mask_img))

        return image1, mask_class, cat_label


def load_transformed_dataset():
    # 原图变换
    data_transforms = [
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(_my_normalization)
    ]
    data_transform = transforms.Compose(data_transforms)

    # 掩码变换
    mask_transform = get_mask_transforms()

    data_size = None
    TRAIN_IMAGE_DIR = './img_renamed'
    PSEUDO_MASK_DIR = './mask_renamed'
    img1_list = get_images_list(TRAIN_IMAGE_DIR, k=data_size)

    # 划分训练/验证集（保持原图和掩码一一对应）
    ratio = 0.9
    idxs = np.random.RandomState(2023).permutation(img1_list.shape[0])
    split = int(img1_list.shape[0] * ratio)
    train_index = idxs[:split]
    valid_index = idxs[split:]

    train_dataset = Histo_Dataset(
        image1_dir=TRAIN_IMAGE_DIR,
        mask_dir=PSEUDO_MASK_DIR,
        image1_list=img1_list[train_index],
        transform=data_transform,
        mask_transform=mask_transform
    )
    eval_dataset = Histo_Dataset(
        image1_dir=TRAIN_IMAGE_DIR,
        mask_dir=PSEUDO_MASK_DIR,
        image1_list=img1_list[valid_index],
        transform=data_transform,
        mask_transform=mask_transform
    )

    return train_dataset, eval_dataset


def reverse_transforms_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


def get_beta_schedule(betas):
    schedule = {}
    schedule['alphas'] = 1. - betas
    schedule['alphas_cumprod'] = torch.cumprod(schedule['alphas'], dim=0)
    schedule['alphas_cumprod_prev'] = F.pad(schedule['alphas_cumprod'][:-1], (1, 0), value=1.0)
    schedule['sqrt_recip_alphas'] = torch.sqrt(1.0 / schedule['alphas'])
    schedule['sqrt_alphas_cumprod'] = torch.sqrt(schedule['alphas_cumprod'])
    schedule['sqrt_one_minus_alphas_cumprod'] = torch.sqrt(1. - schedule['alphas_cumprod'])
    schedule['posterior_variance'] = betas * (1. - schedule['alphas_cumprod_prev']) / (
            1. - schedule['alphas_cumprod'])
    return schedule


# 修改：融合语义引导损失
def get_loss(noise, noise_pred, t, betas_schedule, gpu,
             mask=None, cat_label=None, noise_sem_pred=None):
    # 1. 原始噪声预测损失
    t_cpu = t.cpu()
    snr = 1.0 / (1 - betas_schedule['alphas_cumprod'][t_cpu]) - 1
    k = 1.0
    gamma = 1.0
    lambda_t = 1.0 / ((k + snr) ** gamma)
    lambda_t = lambda_t.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(gpu)

    n = noise.shape[1] * noise.shape[2] * noise.shape[3]
    loss_noise = torch.sum(lambda_t * F.mse_loss(noise, noise_pred, reduction='none')) / n

    # 2. 语义引导损失
    loss_sem = 0.0
    if mask is not None and noise_sem_pred is not None:
        # 2.1 空间语义损失（掩码类别分组MSE）
        batch_size = mask.shape[0]
        sem_loss_list = []
        for b in range(batch_size):
            for cls in range(5):
                cls_mask = (mask[b] == cls).float()
                if cls_mask.sum() == 0:
                    continue
                noise_cls = noise[b, :, mask[b] == cls]
                noise_pred_cls = noise_sem_pred[b, :, mask[b] == cls]
                sem_loss_list.append(F.mse_loss(noise_pred_cls, noise_cls))

        loss_sem_spatial = torch.mean(torch.stack(sem_loss_list)) if sem_loss_list else 0.0

        # 2.2 全局语义损失（类别标签余弦相似度）
        if cat_label is not None:
            noise_mean = noise_pred.mean(dim=(2, 3))
            cat_label_exp = cat_label.unsqueeze(1).repeat(1, 3, 1)
            loss_sem_global = 1 - F.cosine_similarity(
                noise_mean.unsqueeze(2), cat_label_exp, dim=1
            ).mean()
        else:
            loss_sem_global = 0.0

        loss_sem = 0.3 * loss_sem_spatial + 0.1 * loss_sem_global

    # 总损失
    total_loss = loss_noise + loss_sem
    return total_loss


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch=None, ddp=False):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, ddp)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, ddp)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, ddp=False):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if epoch != None:
            weight_path = self.path[:-4] + '_' + str(epoch) + '_' + str(val_loss)[:7] + '.pth'
        else:
            weight_path = self.path
        if ddp:
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'model_state_dict': model_state,
        }, weight_path)
        self.val_loss_min = val_loss