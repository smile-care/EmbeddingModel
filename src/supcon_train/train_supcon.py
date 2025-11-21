"""
SupCon监督对比学习训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
from typing import Optional

from .models.supcon_model import SupConModel
from .models.losses import CombinedLoss
from .datasets.supcon_dataset import SupConDataset
from ..utils.config_loader import load_config
from ..utils.logging import setup_logger
from ..utils.visualization import plot_loss_curve


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_classification_head: bool = False
) -> dict:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_supcon_loss = 0.0
    total_ce_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # 获取数据（使用view1）
        images = batch['view1'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(images, return_features=False)
        embeddings = outputs['embeddings']
        
        if use_classification_head:
            logits = outputs['logits']
            loss, supcon_loss, ce_loss = criterion(embeddings, logits, labels)
        else:
            # 只使用SupCon loss
            from .models.losses import SupervisedContrastiveLoss
            supcon_criterion = SupervisedContrastiveLoss()
            loss = supcon_criterion(embeddings, labels)
            supcon_loss = loss.item()
            ce_loss = 0.0
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_supcon_loss += supcon_loss if isinstance(supcon_loss, float) else supcon_loss.item()
        total_ce_loss += ce_loss if isinstance(ce_loss, float) else ce_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': loss.item(),
            'supcon': supcon_loss if isinstance(supcon_loss, float) else supcon_loss.item(),
            'ce': ce_loss if isinstance(ce_loss, float) else ce_loss.item()
        })
    
    return {
        'loss': total_loss / num_batches,
        'supcon_loss': total_supcon_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_classification_head: bool = False
) -> dict:
    """验证"""
    model.eval()
    total_loss = 0.0
    total_supcon_loss = 0.0
    total_ce_loss = 0.0
    num_batches = 0
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['view1'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, return_features=False)
            embeddings = outputs['embeddings']
            
            if use_classification_head:
                logits = outputs['logits']
                loss, supcon_loss, ce_loss = criterion(embeddings, logits, labels)
                
                # 计算准确率
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else:
                from .models.losses import SupervisedContrastiveLoss
                supcon_criterion = SupervisedContrastiveLoss()
                loss = supcon_criterion(embeddings, labels)
                supcon_loss = loss.item()
                ce_loss = 0.0
            
            total_loss += loss.item()
            total_supcon_loss += supcon_loss if isinstance(supcon_loss, float) else supcon_loss.item()
            total_ce_loss += ce_loss if isinstance(ce_loss, float) else ce_loss.item()
            num_batches += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return {
        'loss': total_loss / num_batches,
        'supcon_loss': total_supcon_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='SupCon监督对比学习训练')
    parser.add_argument('--config', type=str, default='configs/supcon_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    args = parser.parse_args()
    
    # 加载配置
    supcon_config = load_config(args.config)
    data_config = load_config(args.data_config)
    
    # 设置日志
    logger = setup_logger(
        'supcon_train',
        log_dir=supcon_config['supcon']['output']['log_dir']
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    checkpoint_dir = Path(supcon_config['supcon']['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化wandb
    if supcon_config['supcon']['output'].get('use_wandb', False):
        if wandb is None:
            logger.warning("wandb not installed, skipping wandb logging")
        else:
            wandb.init(
                project=supcon_config['supcon']['output'].get('wandb_project', 'industrial-supcon'),
                config=supcon_config['supcon']
            )
    
    # 创建数据集
    logger.info("加载数据集...")
    supcon_metadata = data_config['data']['metadata_root'] + "/supcon_dataset.json"
    dataset = SupConDataset(
        metadata_file=supcon_metadata,
        patch_root=data_config['data']['patches_root'],
        augmentation_config=supcon_config['supcon']['augmentation']
    )
    
    # 获取类别数
    num_classes = len(dataset.labels)
    logger.info(f"类别数: {num_classes}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=supcon_config['supcon']['data']['batch_size'],
        shuffle=True,
        num_workers=supcon_config['supcon']['data']['num_workers'],
        pin_memory=supcon_config['supcon']['data']['pin_memory']
    )
    
    # 创建模型
    logger.info("创建模型...")
    model_config = supcon_config['supcon']['model']
    model = SupConModel(
        backbone_type=model_config['backbone'],
        backbone_checkpoint=model_config.get('ssl_checkpoint'),
        embedding_dim=model_config['embedding_dim'],
        projection_hidden_dims=model_config['projection_head']['hidden_dims'],
        num_classes=num_classes if model_config.get('use_classification_head') else None,
        image_size=224,
        freeze_backbone=supcon_config['supcon']['training_strategy'].get('freeze_backbone_epochs', 0) > 0
    ).to(device)
    
    use_classification_head = model_config.get('use_classification_head', False)
    
    # 创建Loss
    loss_config = supcon_config['supcon']['loss']
    if use_classification_head:
        criterion = CombinedLoss(
            supcon_weight=loss_config['supcon']['weight'],
            ce_weight=loss_config['cross_entropy']['weight'],
            temperature=loss_config['supcon']['temperature'],
            label_smoothing=loss_config['cross_entropy']['label_smoothing']
        )
    else:
        from .models.losses import SupervisedContrastiveLoss
        criterion = SupervisedContrastiveLoss(
            temperature=loss_config['supcon']['temperature']
        )
    
    # 创建优化器（backbone和projection head使用不同学习率）
    training_config = supcon_config['supcon']['training']
    backbone_lr_ratio = training_config.get('backbone_lr_ratio', 0.1)
    
    # 分离backbone和projection head的参数
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': training_config['learning_rate'] * backbone_lr_ratio},
        {'params': other_params, 'lr': training_config['learning_rate']}
    ], weight_decay=training_config['weight_decay'])
    
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
    
    # 训练策略：逐步解冻backbone
    training_strategy = supcon_config['supcon']['training_strategy']
    freeze_epochs = training_strategy.get('freeze_backbone_epochs', 0)
    
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
        # 解冻backbone（如果需要）
        if epoch == freeze_epochs and freeze_epochs > 0:
            logger.info("解冻backbone参数...")
            model.unfreeze_all()
        
        # 训练
        train_metrics = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch,
            use_classification_head
        )
        train_losses.append(train_metrics['loss'])
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        val_metrics = validate(
            model, dataloader, criterion, device, use_classification_head
        )
        val_losses.append(val_metrics['loss'])
        
        # 记录日志
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics.get('accuracy', 0):.2f}%, "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Wandb记录
        if supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_supcon_loss': train_metrics['supcon_loss'],
                'val_loss': val_metrics['loss'],
                'val_supcon_loss': val_metrics['supcon_loss'],
                'learning_rate': scheduler.get_last_lr()[0]
            }
            if use_classification_head:
                log_dict.update({
                    'train_ce_loss': train_metrics['ce_loss'],
                    'val_ce_loss': val_metrics['ce_loss'],
                    'val_accuracy': val_metrics.get('accuracy', 0)
                })
            wandb.log(log_dict)
        
        # 保存checkpoint
        if epoch % training_config.get('save_interval', 5) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'best_loss': best_loss,
                'config': supcon_config
            }
            torch.save(
                checkpoint,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            )
        
        # 保存最佳模型
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'config': supcon_config,
                    'num_classes': num_classes
                },
                checkpoint_dir / "best_model.pth"
            )
            logger.info(f"保存最佳模型 (val_loss={best_loss:.4f})")
    
    # 绘制损失曲线
    plot_loss_curve(
        train_losses,
        val_losses,
        save_path=str(checkpoint_dir / "loss_curve.png"),
        title="SupCon Training Loss"
    )
    
    logger.info("训练完成！")
    if supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    main()

