"""
可视化工具
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch


def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Loss"
):
    """
    绘制损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表（可选）
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses:
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
    n_samples: int = 8
):
    """
    可视化MAE重建结果
    
    Args:
        original: 原始图像 (B, C, H, W)
        reconstructed: 重建图像 (B, C, H, W)
        masked: 带mask的图像 (B, C, H, W)
        save_path: 保存路径
        n_samples: 显示的样本数
    """
    n_samples = min(n_samples, original.size(0))
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
    
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

