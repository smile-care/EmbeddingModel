"""
SupCon监督对比学习训练脚本
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, DataLoader, Sampler, WeightedRandomSampler
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None
from typing import Optional

from ..utils.config_loader import load_config
from ..utils.logging import setup_logger
from ..utils.visualization import plot_loss_curve
from .datasets.supcon_dataset import SupConDataset
from .models.losses import CombinedLoss
from .models.supcon_model import SupConModel


class LabelBalancedBatchSampler(Sampler):
    """
    确保每个batch中每个类别都有多个样本的BatchSampler
    这对于SupCon训练很重要，因为需要positive pairs
    """
    def __init__(self, dataset, batch_size, samples_per_class=2, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last
        
        # 按label组织样本索引
        self.label_indices = defaultdict(list)
        for idx, item in enumerate(dataset.patches):
            label = item['label']
            self.label_indices[label].append(idx)
        
        self.labels = list(self.label_indices.keys())
        self.num_classes = len(self.labels)
        
        # 计算每个batch可以包含多少个类别
        # 每个类别samples_per_class个样本
        self.classes_per_batch = min(
            self.num_classes,
            batch_size // samples_per_class
        )
        
        # 计算实际batch大小（可能小于配置的batch_size）
        self.actual_batch_size = self.classes_per_batch * samples_per_class
        
    def __iter__(self):
        # 为每个epoch生成batch
        # 打乱每个类别的索引
        for label in self.labels:
            random.shuffle(self.label_indices[label])
        
        # 创建循环迭代器（当样本用完时重新开始）
        label_iterators = {}
        for label in self.labels:
            label_iterators[label] = self._cycle_iterator(self.label_indices[label])
        
        # 生成多个batch
        num_batches = len(self.dataset) // self.actual_batch_size
        if not self.drop_last:
            num_batches += 1
        
        for _ in range(num_batches):
            batch_indices = []
            selected_labels = random.sample(self.labels, min(self.classes_per_batch, len(self.labels)))
            
            for label in selected_labels:
                # 为每个类别采样samples_per_class个样本
                for _ in range(self.samples_per_class):
                    idx = next(label_iterators[label])
                    batch_indices.append(idx)
            
            # 打乱batch内的顺序
            random.shuffle(batch_indices)
            yield batch_indices
    
    def _cycle_iterator(self, items):
        """创建一个循环迭代器"""
        while True:
            for item in items:
                yield item
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.actual_batch_size
        else:
            return (len(self.dataset) + self.actual_batch_size - 1) // self.actual_batch_size



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
        
        # 检查embeddings是否包含NaN或Inf
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            print(f"警告：Epoch {epoch}, Batch {num_batches}: embeddings包含NaN或Inf，跳过此batch")
            continue
        
        if use_classification_head:
            logits = outputs['logits']
            # 检查logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"警告：Epoch {epoch}, Batch {num_batches}: logits包含NaN或Inf，跳过此batch")
                continue
            loss, supcon_loss, ce_loss = criterion(embeddings, logits, labels)
        else:
            # 只使用SupCon loss
            from .models.losses import SupervisedContrastiveLoss
            supcon_criterion = SupervisedContrastiveLoss()
            loss = supcon_criterion(embeddings, labels)
            supcon_loss = loss.item()
            ce_loss = 0.0
        
        # 检查loss是否为NaN或Inf
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
            print(f"警告：Epoch {epoch}, Batch {num_batches}: loss为NaN或Inf，跳过此batch")
            continue
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
            
            # 检查embeddings是否包含NaN或Inf
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print(f"警告：验证时embeddings包含NaN或Inf，跳过此batch")
                continue
            
            if use_classification_head:
                logits = outputs['logits']
                # 检查logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"警告：验证时logits包含NaN或Inf，跳过此batch")
                    continue
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
            
            # 检查loss是否为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
                print(f"警告：验证时loss为NaN或Inf，跳过此batch")
                continue
            
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
    
    # 获取数据集路径（优先从supcon配置读取，否则使用data_config）
    data_cfg = supcon_config['supcon']['data']
    
    # 支持多个数据集文件
    if 'train_metadata_files' in data_cfg:
        # 支持多个训练集文件
        train_metadata = data_cfg['train_metadata_files']
        if isinstance(train_metadata, str):
            train_metadata = [train_metadata]
        logger.info(f"训练集（多个文件）: {train_metadata}")
    elif 'train_metadata' in data_cfg:
        # 单个训练集文件
        train_metadata = data_cfg['train_metadata']
        logger.info(f"训练集: {train_metadata}")
    else:
        # 兼容旧配置：使用同一个数据集
        train_metadata = data_config['data']['metadata_root'] + "/supcon_dataset.json"
        logger.info(f"训练集（默认）: {train_metadata}")
    
    if 'val_metadata_files' in data_cfg:
        # 支持多个验证集文件
        val_metadata = data_cfg['val_metadata_files']
        if isinstance(val_metadata, str):
            val_metadata = [val_metadata]
        logger.info(f"验证集（多个文件）: {val_metadata}")
    elif 'val_metadata' in data_cfg:
        # 单个验证集文件
        val_metadata = data_cfg['val_metadata']
        logger.info(f"验证集: {val_metadata}")
    else:
        # 使用训练集作为验证集
        val_metadata = train_metadata
        logger.info(f"验证集（使用训练集）: {val_metadata}")
    
    train_dataset = SupConDataset(
        metadata_file=train_metadata,
        patch_root=None,  # 不再需要patch_root，因为现在使用绝对路径
        augmentation_config=supcon_config['supcon']['augmentation']
    )
    
    val_dataset = SupConDataset(
        metadata_file=val_metadata,
        patch_root=None,  # 不再需要patch_root，因为现在使用绝对路径
        augmentation_config=None  # 验证集不使用数据增强
    )
    
    # 获取类别数
    num_classes = len(train_dataset.labels)
    logger.info(f"训练集样本数: {len(train_dataset)}, 类别数: {num_classes}")
    logger.info(f"验证集样本数: {len(val_dataset)}")
    
    # 创建数据加载器
    # 对于SupCon，需要确保每个batch中有相同标签的样本
    batch_size = supcon_config['supcon']['data']['batch_size']
    use_label_balanced = supcon_config['supcon']['data'].get('samples_per_class', 0) > 0
    
    if use_label_balanced:
        # 使用label balanced batch sampler，确保每个batch中每个类别有多个样本
        samples_per_class = supcon_config['supcon']['data']['samples_per_class']
        logger.info(f"使用label balanced batch sampler，每个batch中每个类别至少{samples_per_class}个样本")
        
        # 创建训练集LabelBalancedBatchSampler
        train_batch_sampler = LabelBalancedBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
            drop_last=False
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=supcon_config['supcon']['data']['num_workers'],
            pin_memory=supcon_config['supcon']['data']['pin_memory']
        )
        
        logger.info(f"训练集实际batch大小: {train_batch_sampler.actual_batch_size} (配置: {batch_size})")
    else:
        # 使用普通shuffle
        logger.warning("未使用label balanced采样，某些batch可能没有positive pairs！")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=supcon_config['supcon']['data']['num_workers'],
            pin_memory=supcon_config['supcon']['data']['pin_memory']
        )
    
    # 验证集dataloader（不需要label balanced，使用普通顺序加载）
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
        {'params': backbone_params, 'lr': float(training_config['learning_rate']) * backbone_lr_ratio},
        {'params': other_params, 'lr': float(training_config['learning_rate'])}
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
        # 解冻backbone（如果需要）
        if epoch == freeze_epochs and freeze_epochs > 0:
            logger.info("解冻backbone参数...")
            model.unfreeze_all()
        
        # 训练
        train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch,
            use_classification_head
        )
        train_losses.append(train_metrics['loss'])
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        val_metrics = validate(
            model, val_dataloader, criterion, device, use_classification_head
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

