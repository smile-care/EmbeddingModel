"""
可视化工具
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    反归一化图像（从ImageNet归一化恢复到[0,1]范围）
    
    Args:
        tensor: 归一化后的图像tensor (C, H, W) 或 (B, C, H, W)
        mean: 归一化均值
        std: 归一化标准差
        
    Returns:
        反归一化后的tensor，值域在[0, 1]
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:  # (B, C, H, W)
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    if tensor.device.type == 'cuda':
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    # 反归一化: (x * std) + mean
    denorm = tensor * std + mean
    # 裁剪到[0, 1]范围
    denorm = torch.clamp(denorm, 0, 1)
    return denorm


def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    val_epochs: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Loss"
):
    """
    绘制损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表（可选）
        val_epochs: 验证对应的 epoch 编号（与 val_losses 等长；缺省则按 1..N 绘制）
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        if val_epochs and len(val_epochs) == len(val_losses):
            plt.plot(val_epochs, val_losses, label='Val Loss', linewidth=2, marker='o')
        else:
            plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    labels: List[str],
    save_path: Optional[str] = None,
    title: str = "Distance Matrix"
):
    """
    绘制距离热力图
    
    Args:
        distance_matrix: 距离矩阵 (N x N)
        labels: 类别标签列表
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        distance_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='.3f',
        cmap='viridis_r',
        square=True,
        cbar_kws={'label': 'Distance'}
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    masked: torch.Tensor,
    save_path: Optional[str] = None,
    n_samples: int = 8,
    denormalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
):
    """
    可视化MAE重建结果
    
    Args:
        original: 原始图像 (B, C, H, W)，如果已归一化需要denormalize=True
        reconstructed: 重建图像 (B, C, H, W)，通常是原始像素值[0,1]
        masked: 带mask的图像 (B, C, H, W)，如果已归一化需要denormalize=True
        save_path: 保存路径
        n_samples: 显示的样本数
        denormalize: 是否对原始和masked图像进行反归一化
        mean: 归一化均值（如果denormalize=True）
        std: 归一化标准差（如果denormalize=True）
    """
    n_samples = min(n_samples, original.size(0))
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
    
    # 反归一化原始图像和masked图像（如果已归一化）
    if denormalize:
        original = denormalize_image(original, mean, std)
        masked = denormalize_image(masked, mean, std)
    
    # 确保重建图像在[0, 1]范围内
    # reconstructed = torch.clamp(reconstructed, 0, 1)
    if denormalize:
        reconstructed = denormalize_image(reconstructed, mean, std)
    
    for i in range(n_samples):
        # 原始图像
        img_orig = original[i].permute(1, 2, 0).cpu().numpy()
        img_orig = np.clip(img_orig, 0, 1)
        axes[0, i].imshow(img_orig)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Masked图像
        img_masked = masked[i].permute(1, 2, 0).cpu().numpy()
        img_masked = np.clip(img_masked, 0, 1)
        axes[1, i].imshow(img_masked)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Masked', fontsize=10)
        
        # 重建图像
        img_recon = reconstructed[i].permute(1, 2, 0).cpu().numpy()
        img_recon = np.clip(img_recon, 0, 1)
        axes[2, i].imshow(img_recon)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

