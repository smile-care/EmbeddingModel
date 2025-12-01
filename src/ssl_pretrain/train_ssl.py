"""
MAE自监督预训练脚本
"""
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None
from typing import Optional

from ..utils.config_loader import load_config
from ..utils.logging import setup_logger
from ..utils.visualization import plot_loss_curve, visualize_reconstruction
from .datasets.ssl_dataset import SSLDataset
from .models.mae import MAE


def _create_masked_visualization(
    images: torch.Tensor,
    masks: torch.Tensor,
    patch_size: int = 16,
    image_size: int = 224
) -> torch.Tensor:
    """
    根据mask创建masked图像的可视化
    
    Args:
        images: 原始图像 (B, C, H, W)
        masks: mask (B, N)，1表示masked，0表示可见
        patch_size: patch大小
        image_size: 图像大小
        
    Returns:
        masked图像 (B, C, H, W)，masked区域用黑色填充
    """
    B, C, H, W = images.shape
    h_patches = w_patches = image_size // patch_size
    
    # 将mask从(B, N)转换为(B, h_patches, w_patches)
    mask_2d = masks.view(B, h_patches, w_patches)
    
    # 扩展mask到patch大小：使用repeat_interleave更简单
    # (B, h_patches, w_patches) -> (B, H, W)
    mask_2d = mask_2d.unsqueeze(1)  # (B, 1, h_patches, w_patches)
    # 在每个维度上重复patch_size次
    mask_2d = mask_2d.repeat_interleave(patch_size, dim=2)  # (B, 1, H, w_patches)
    mask_2d = mask_2d.repeat_interleave(patch_size, dim=3)  # (B, 1, H, W)
    
    # 创建masked图像：masked区域用黑色填充
    masked_imgs = images.clone()
    masked_imgs = masked_imgs * (1 - mask_2d)  # masked区域变为0（黑色）
    
    return masked_imgs


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        
        # 前向传播
        loss, pred_img, mask = model(images)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Optional[Path] = None,
    epoch: int = 0
) -> float:
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    sample_images = []
    sample_preds = []
    sample_masks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            
            loss, pred_img, mask = model(images)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 保存样本用于可视化
            if len(sample_images) < 8:
                sample_images.append(images[:1].cpu())
                sample_preds.append(pred_img[:1].cpu())
                sample_masks.append(mask[:1].cpu())
    
    avg_loss = total_loss / num_batches
    
    # 可视化重建结果
    if save_dir and len(sample_images) > 0:
        orig = torch.cat(sample_images, dim=0)
        recon = torch.cat(sample_preds, dim=0)
        masks = torch.cat(sample_masks, dim=0)
        
        # 从模型中获取patch_size和image_size
        if hasattr(model, 'patch_size') and hasattr(model, 'image_size'):
            patch_size = model.patch_size
            image_size = model.image_size
        else:
            # 默认值
            patch_size = 16
            image_size = 224
        
        # 创建masked图像用于可视化
        # mask形状是(B, N)，需要转换为(B, C, H, W)的图像mask
        masked_imgs = _create_masked_visualization(
            orig, masks, patch_size=patch_size, image_size=image_size
        )
        
        visualize_reconstruction(
            orig, recon, masked_imgs,
            save_path=str(save_dir / f"reconstruction_epoch_{epoch}.png"),
            n_samples=len(sample_images),
            denormalize=True  # 启用反归一化，因为输入图像已归一化
        )
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='MAE自监督预训练')
    parser.add_argument('--config', type=str, default='configs/ssl_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    args = parser.parse_args()
    
    # 加载配置
    ssl_config = load_config(args.config)
    data_config = load_config(args.data_config)
    
    # 设置日志
    logger = setup_logger(
        'ssl_train',
        log_dir=ssl_config['ssl']['output']['log_dir']
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    checkpoint_dir = Path(ssl_config['ssl']['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化wandb
    if ssl_config['ssl']['output'].get('use_wandb', False):
        if wandb is None:
            logger.warning("wandb not installed, skipping wandb logging")
        else:
            wandb.init(
                project=ssl_config['ssl']['output'].get('wandb_project', 'industrial-ssl'),
                config=ssl_config['ssl']
            )
    
    # 创建数据集
    logger.info("加载数据集...")
    
    # 支持从配置中读取多个数据集文件，或使用默认路径
    data_cfg = ssl_config['ssl']['data']
    if 'metadata_files' in data_cfg:
        # 支持多个数据集文件
        metadata_files = data_cfg['metadata_files']
        if isinstance(metadata_files, str):
            metadata_files = [metadata_files]
        logger.info(f"加载多个数据集: {metadata_files}")
    else:
        # 兼容旧配置：使用单个数据集文件
        metadata_files = [data_config['data']['metadata_root'] + "/ssl_dataset.json"]
        logger.info(f"加载单个数据集: {metadata_files[0]}")
    
    dataset = SSLDataset(
        metadata_file=metadata_files,
        image_root=None,  # 不再需要image_root，因为现在使用绝对路径
        augmentation_config=ssl_config['ssl']['augmentation'],
        domain_balanced=ssl_config['ssl']['data']['domain_balanced']
    )
    
    logger.info(f"数据集加载完成: 总样本数={len(dataset)}, domains={len(set(img['domain_id'] for img in dataset.images))}")
    
    # 创建数据加载器
    sampler = dataset.get_domain_balanced_sampler() if ssl_config['ssl']['data']['domain_balanced'] else None
    dataloader = DataLoader(
        dataset,
        batch_size=ssl_config['ssl']['data']['batch_size'],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=ssl_config['ssl']['data']['num_workers'],
        pin_memory=ssl_config['ssl']['data']['pin_memory']
    )
    
    # 创建模型
    logger.info("创建模型...")
    model_config = ssl_config['ssl']['model']
    model = MAE(
        backbone_type=model_config['backbone'],
        image_size=model_config['image_size'],
        patch_size=model_config['patch_size'],
        mask_ratio=model_config['mask_ratio'],
        decoder_dim=model_config['decoder_dim'],
        decoder_depth=model_config['decoder_depth'],
        decoder_num_heads=model_config['decoder_num_heads'],
        backbone_pretrained=model_config['backbone_pretrained']
    ).to(device)
    
    # 创建优化器
    training_config = ssl_config['ssl']['training']
    if training_config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(training_config['learning_rate']),
            weight_decay=training_config['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
    
    # 学习率调度器
    if training_config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['epochs'],
            eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config['epochs'] // 3,
            gamma=0.1
        )
    
    # 恢复训练
    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        logger.info(f"从checkpoint恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
    
    # 训练循环
    logger.info("开始训练...")
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, training_config['epochs']+1):
        # 训练
        train_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 验证（如果有验证集）
        val_loss = train_loss  # 简化：使用训练loss
        if epoch % training_config.get('eval_interval', 10) == 0 or epoch == 1:
            val_loss = validate(
                model, dataloader, device,
                save_dir=checkpoint_dir,
                epoch=epoch
            )
            val_losses.append(val_loss)
        
        # 记录日志
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Wandb记录
        if ssl_config['ssl']['output'].get('use_wandb', False) and wandb is not None:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        # 保存checkpoint
        if epoch % training_config.get('save_interval', 10) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': best_loss,
                'config': ssl_config
            }
            torch.save(
                checkpoint,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            )
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'config': ssl_config
                },
                checkpoint_dir / "best_model.pth"
            )
            logger.info(f"保存最佳模型 (val_loss={best_loss:.4f})")
    
    # 绘制损失曲线
    plot_loss_curve(
        train_losses,
        val_losses if val_losses else None,
        save_path=str(checkpoint_dir / "loss_curve.png"),
        title="MAE Training Loss"
    )
    
    logger.info("训练完成！")
    if ssl_config['ssl']['output'].get('use_wandb', False) and wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    main()

