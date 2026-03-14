import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import utils
from model import DiffusionNet

import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import numpy as np
from scipy.io import savemat
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 单GPU设置
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
EPOCHS = 150
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
T = 1000
DATA_TYPE = 'luad_withpretrainmask_1.24'

betas = utils.quadratic_beta_schedule(timesteps=T)
betas_schedule = utils.get_beta_schedule(betas)

@torch.no_grad()
def sample_timestep(model, x, t):
    """单卡版本的时间步采样"""
    betas_t = utils.get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = utils.get_index_from_list(
        betas_schedule['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = utils.get_index_from_list(betas_schedule['sqrt_recip_alphas'], t, x.shape)

    # 关键修改1：传入完整4通道x（而非仅前3通道），匹配模型输入要求
    # 关键修改2：构造全0的类别标签（采样时无真实类别，补全参数）
    batch_size = x.shape[0]
    cat_label = torch.zeros((batch_size, 4), device=DEVICE)  # 4维类别标签全0

    # 核心修复：模型输出3通道噪声 → 扩展为4通道（掩码通道补0）
    noise_pred_3ch = model(x, t, cat_label)  # 模型输出3通道（原图噪声）
    # 拼接0填充的第4通道（掩码通道），匹配x的4通道维度
    noise_pred_4ch = torch.cat([
        noise_pred_3ch,
        torch.zeros_like(x[:, 3:4, :, :])  # 第4通道（掩码）补0
    ], dim=1)

    # 用扩展后的4通道噪声预测计算model_mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred_4ch / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = utils.get_index_from_list(betas_schedule['posterior_variance'], t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image(model, epoch):
    """单卡版本的图像采样与保存"""
    # 采样时仅生成3通道原图（掩码为0）
    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
    mask = torch.zeros((1, 1, IMG_SIZE, IMG_SIZE), device=DEVICE)
    x = torch.cat([img, mask], dim=1)  # 拼接为4通道输入
    num_images = 100
    stepsize = int(T / num_images)
    all_images = []

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
        x = sample_timestep(model, x, t)
        if i % stepsize == 0:
            all_images.append(x[:, :3, :, :])  # 仅保存原图3通道

    fig, axs = plt.subplots(10, 10)
    x_idx = 0
    for i in range(10):
        for j in range(10):
            out_img = utils.reverse_transforms_image(all_images[x_idx].detach().cpu())
            axs[i, j].imshow(out_img)
            axs[i, j].axis('off')
            x_idx += 1
    plt.savefig(f'./images/{DATA_TYPE}/image_{epoch}.png', dpi=300)
    plt.close()

def initialize_weights(model):
    """模型权重初始化"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight.data, 0.0, 0.01)

def train_epoch(train_dataloader, model, optimizer, epoch):
    """单卡训练轮次"""
    model.train()
    losses = []
    p_bar = tqdm(train_dataloader)

    for batch in p_bar:
        optimizer.zero_grad()
        # 解包：原图、掩码、类别标签
        img_batch, mask_batch, cat_batch = batch
        img_batch = img_batch.to(DEVICE)
        # 关键修改：给mask增加通道维度（从[B,H,W]→[B,1,H,W]）
        mask_batch = mask_batch.unsqueeze(1).to(DEVICE)
        cat_batch = cat_batch.to(DEVICE)

        # 拼接原图+掩码为4通道输入（此时img_batch[B,3,H,W] + mask_batch[B,1,H,W] → [B,4,H,W]）
        x_input = torch.cat([img_batch, mask_batch], dim=1)

        # 随机时间步
        t = torch.randint(0, T, (img_batch.shape[0],)).long().to(DEVICE)
        # 前向扩散（仅对原图3通道）
        x_noisy, noise = utils.forward_diffusion_sample(img_batch, t, betas_schedule, DEVICE)
        # 拼接噪声图+掩码为4通道输入
        x_noisy_input = torch.cat([x_noisy, mask_batch], dim=1)

        # 模型预测噪声
        noise_pred = model(x_noisy_input, t, cat_batch)

        # 计算损失（加入掩码和类别标签加权）
        loss = utils.get_loss(
            noise, noise_pred, t, betas_schedule,
            DEVICE, mask_batch.squeeze(1), cat_batch  # 损失计算时还原掩码维度（去掉通道维）
        )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        p_bar.set_description(f'Epoch {epoch}')
        p_bar.set_postfix(loss=loss.item())

    avg_loss = np.mean(losses)
    print(f'Epoch: {epoch}\ttrain_loss: {avg_loss:.4f}')
    return avg_loss

def eval_epoch(eval_dataloader, model, epoch):
    """单卡验证轮次"""
    model.eval()
    losses = []
    p_bar = tqdm(eval_dataloader)

    with torch.no_grad():
        for batch in p_bar:
            img_batch, mask_batch, cat_batch = batch
            img_batch = img_batch.to(DEVICE)
            # 关键修改：给mask增加通道维度
            mask_batch = mask_batch.unsqueeze(1).to(DEVICE)
            cat_batch = cat_batch.to(DEVICE)

            # 拼接输入
            x_input = torch.cat([img_batch, mask_batch], dim=1)
            t = torch.randint(0, T, (img_batch.shape[0],)).long().to(DEVICE)
            x_noisy, noise = utils.forward_diffusion_sample(img_batch, t, betas_schedule, DEVICE)
            x_noisy_input = torch.cat([x_noisy, mask_batch], dim=1)

            noise_pred = model(x_noisy_input, t, cat_batch)
            loss = utils.get_loss(
                noise, noise_pred, t, betas_schedule,
                DEVICE, mask_batch.squeeze(1), cat_batch  # 损失计算时还原掩码维度
            )

            losses.append(loss.item())
            p_bar.set_description(f'Epoch {epoch}')
            p_bar.set_postfix(loss=loss.item())

    avg_loss = np.mean(losses)
    print(f'Epoch: {epoch}\teval_loss: {avg_loss:.4f}')
    return avg_loss

def main():
    # 创建保存目录
    checkpoint_path = f'./snapshots/{DATA_TYPE}/'
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(f'./images/{DATA_TYPE}', exist_ok=True)
    os.makedirs('./plots/diff', exist_ok=True)

    # 加载数据集（含原图+掩码+类别标签）
    train_dataset, eval_dataset = utils.load_transformed_dataset()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型（4通道输入）
    model = DiffusionNet(dim=64, channels=4).to(DEVICE)
    initialize_weights(model)
    print(f"Num params: {sum(p.numel() for p in model.parameters())}")

    # 优化器
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 加载 checkpoint
    load_from_chkpt = None
    if load_from_chkpt is not None:
        print(f'Loading checkpoint from: {load_from_chkpt}')
        checkpoint = torch.load(load_from_chkpt, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 早停机制
    early_stopping = utils.EarlyStopping(
        patience=20,
        verbose=True,
        path=f'{checkpoint_path}{BATCH_SIZE}_{LEARNING_RATE}.pth'
    )

    # 训练循环
    train_losses = []
    eval_losses = []
    start_time = time.process_time()

    for epoch in range(EPOCHS):
        print(f'epoch {epoch + 1}/{EPOCHS}')
        train_loss = train_epoch(train_dataloader, model, optimizer, epoch + 1)
        eval_loss = eval_epoch(eval_dataloader, model, epoch + 1)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        # 每20轮采样一次图像
        if (epoch + 1) % 20 == 0:
            sample_plot_image(model, epoch + 1)

        # 早停检查
        early_stopping(eval_loss, model, epoch + 1)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 训练时间统计
    current_time = time.process_time()
    print(f"Total Time Elapsed: {current_time - start_time:.5f} seconds")

    # 绘制损失曲线
    epochs = np.arange(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    axes.plot(epochs, train_losses, 'tab:blue', label='Train Loss')
    axes.plot(epochs, eval_losses, 'tab:orange', label='Eval Loss')
    axes.set_title(f'Training and Validation Loss (data size: all)',
                   weight='bold', fontsize=7)
    axes.set_xlabel('Epochs', weight='bold', fontsize=9)
    axes.set_ylabel('Loss', weight='bold', fontsize=9)
    axes.legend()
    plt.savefig(f'./plots/diff/{DATA_TYPE}_loss.jpg', dpi=300)

if __name__ == '__main__':
    main()