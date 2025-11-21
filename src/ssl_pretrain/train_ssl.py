"""
MAE自监督预训练脚本
"""
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
        
        # 创建masked图像用于可视化
        masked_imgs = orig.clone()
        # 这里简化处理，实际应该根据mask恢复masked patches
        
        visualize_reconstruction(
            orig, recon, masked_imgs,
            save_path=str(save_dir / f"reconstruction_epoch_{epoch}.png"),
            n_samples=len(sample_images)
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
    ssl_metadata = data_config['data']['metadata_root'] + "/ssl_dataset.json"
    dataset = SSLDataset(
        metadata_file=ssl_metadata,
        image_root=data_config['data']['raw_data_root'],
        augmentation_config=ssl_config['ssl']['augmentation'],
        domain_balanced=ssl_config['ssl']['data']['domain_balanced']
    )
    
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
            lr=training_config['learning_rate'],
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
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        logger.info(f"从checkpoint恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
    
    # 训练循环
    logger.info("开始训练...")
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, training_config['epochs']):
        # 训练
        train_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 验证（如果有验证集）
        val_loss = train_loss  # 简化：使用训练loss
        if epoch % training_config.get('eval_interval', 5) == 0:
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

