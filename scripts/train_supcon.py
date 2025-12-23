"""
SupCon监督对比学习训练脚本
基于data_config_zhenyu.yaml配置，支持mask和自定义相似度矩阵
"""
import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None
    
# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.supcon_train.datasets.supcon_dataset import MultiConfigDataset, SupConDataset
from src.supcon_train.models.losses import SimilarityTargetLoss, SupervisedContrastiveLoss
from src.supcon_train.models.supcon_model import SupConModel
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.metrics import knn_evaluation, similarity_distribution_stats
from src.utils.visualization import plot_loss_curve

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    nan_embedding_count = 0
    nan_loss_count = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # 获取数据（同时使用view1和view2）
        view1_images = batch['view1_image'].to(device)
        view1_masks = batch['view1_mask'].to(device)
        view2_images = batch['view2_image'].to(device)
        view2_masks = batch['view2_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播：分别计算view1和view2的embedding
        outputs1 = model(view1_images, view1_masks, return_features=False)
        embeddings1 = outputs1['embeddings']
        
        outputs2 = model(view2_images, view2_masks, return_features=False)
        embeddings2 = outputs2['embeddings']
        
        # 拼接view1和view2的embedding，形成2B大小的batch
        # embeddings: [view1_0, view1_1, ..., view1_B-1, view2_0, view2_1, ..., view2_B-1]
        # labels_duplicated: [label_0, label_1, ..., label_B-1, label_0, label_1, ..., label_B-1]
        # 这样同一个样本的view1和view2（索引i和i+B）有相同的label，会被视为positive pair
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # (2B, D)
        labels_duplicated = torch.cat([labels, labels], dim=0)  # (2B,)
        
        # 检查embeddings是否包含NaN或Inf
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            nan_embedding_count += 1
            skipped_batches += 1
            if nan_embedding_count <= 5:  # 只打印前5次警告
                print(f"警告：Epoch {epoch}, Batch {num_batches}: embeddings包含NaN或Inf，跳过此batch")
            continue
        

        # 计算SupCon loss
        # 在loss计算中：
        # - 同一个样本的view1和view2（相同label）会被视为positive pair，它们的embedding会被拉近
        # - 不同样本之间根据相似度矩阵判断是否为positive pair
        loss = criterion(embeddings, labels_duplicated)
        
        # 检查loss是否为NaN或Inf
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
            nan_loss_count += 1
            skipped_batches += 1
            if nan_loss_count <= 5:  # 只打印前5次警告
                print(f"警告：Epoch {epoch}, Batch {num_batches}: loss为NaN或Inf，跳过此batch")
            continue
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': loss.item(),
        })
    
    # 打印统计信息
    if skipped_batches > 0:
        print(f"Epoch {epoch}统计: 跳过{skipped_batches}个batch (NaN embedding: {nan_embedding_count}, NaN loss: {nan_loss_count})")
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'skipped_batches': skipped_batches,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    验证函数，使用相似度分布分析作为评估指标
    
    评估指标（基于整个验证集计算，而非batch平均）：
    - PosSim: 正样本平均相似度（越大越好，通常趋近1）
    - NegSim: 负样本平均相似度（越小越好，接近0或负值）
    - Margin: 正负样本间隔（越大越好，表示判别性越强）
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    nan_embedding_count = 0
    nan_loss_count = 0
    
    # 收集所有验证集的embeddings和labels（用于全局相似度分布分析）
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # 验证时只使用view1（不需要数据增强）
            view1_images = batch['view1_image'].to(device)
            view1_masks = batch['view1_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播：计算embedding
            outputs1 = model(view1_images, view1_masks, return_features=False)
            embeddings1 = outputs1['embeddings']
            
            # 归一化embeddings（L2归一化）
            embeddings1 = F.normalize(embeddings1, dim=1, p=2, eps=1e-8)
            
            # 检查embeddings是否包含NaN或Inf
            if torch.isnan(embeddings1).any() or torch.isinf(embeddings1).any():
                nan_embedding_count += 1
                skipped_batches += 1
                if nan_embedding_count <= 5:  # 只打印前5次警告
                    print(f"警告：验证时embeddings包含NaN或Inf，跳过此batch")
                continue
            
            loss = criterion(embeddings1, labels)
            
            # 检查loss是否为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
                nan_loss_count += 1
                skipped_batches += 1
                if nan_loss_count <= 5:  # 只打印前5次警告
                    print(f"警告：验证时loss为NaN或Inf，跳过此batch")
                continue
            
            # 收集embeddings和labels（用于全局相似度分布分析）
            all_embeddings.append(embeddings1.cpu())  # 移到CPU以节省GPU内存
            all_labels.append(labels.cpu())
            
            total_loss += loss.item()
            num_batches += 1
    
    # 打印统计信息
    if skipped_batches > 0:
        print(f"验证统计: 跳过{skipped_batches}个batch (NaN embedding: {nan_embedding_count}, NaN loss: {nan_loss_count})")
    
        # 基于整个验证集计算相似度分布统计（核心评估指标）
    if len(all_embeddings) > 0:
        # 拼接所有embeddings和labels
        all_embeddings_tensor = torch.cat(all_embeddings, dim=0)  # (N, D)
        all_labels_tensor = torch.cat(all_labels, dim=0)  # (N,)
        
        # 将tensor移回device进行计算
        all_embeddings_tensor = all_embeddings_tensor.to(device)
        all_labels_tensor = all_labels_tensor.to(device)
        
        # 1. 计算相似度分布统计（Margin指标）
        sim_stats = similarity_distribution_stats(all_embeddings_tensor, all_labels_tensor)
        margin_pos_sim = sim_stats['pos_sim']
        margin_neg_sim = sim_stats['neg_sim']
        margin = sim_stats['margin']
        
        # 2. 计算kNN评估指标
        knn_stats = knn_evaluation(all_embeddings_tensor, all_labels_tensor, k=10)
        knn_accuracy = knn_stats.get('knn_accuracy', 0.0)
    else:
        margin_pos_sim = 0.0
        margin_neg_sim = 0.0
        margin = 0.0
        knn_accuracy = 0.0
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        # 相似度分布统计（Margin指标）
        'margin_pos_sim': margin_pos_sim,
        'margin_neg_sim': margin_neg_sim,
        'margin': margin,
        # kNN评估指标（独立指标）
        'knn_accuracy': knn_accuracy,
        'skipped_batches': skipped_batches,
    }



def main():
    parser = argparse.ArgumentParser(description='SupCon监督对比学习训练')
    parser.add_argument('--config', type=str, default='configs/supcon_config.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--data_config', type=str, nargs='+', 
                       default=[
                           'configs/data_config_zhenyu.yaml',
                        #    'configs/data_config_mvtec_ad.yaml'
                        ],
                       help='数据配置文件路径（可以指定多个，用空格分隔）')
    parser.add_argument('--use_eval', action='store_true', default=True, help='是否进行验证')
    parser.add_argument('--no_eval', dest='use_eval', action='store_false', help='禁用验证')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    args = parser.parse_args()
    
    # 加载配置
    supcon_config = load_config(args.config)
    data_config_paths = args.data_config
    
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
    logger.info(f"数据配置文件: {data_config_paths}")
    
    # 如果只有一个配置文件，直接使用 SupConDataset；否则使用 MultiConfigDataset
    if len(data_config_paths) == 1:
        train_dataset = SupConDataset(
            data_config_path=data_config_paths[0],
            split='train',
            augmentation_config=supcon_config['supcon']['augmentation']
        )
        num_classes = len(train_dataset.categories)
        similarity_matrix = train_dataset.get_similarity_matrix()
        default_similarity = train_dataset.default_similarity
    else:
        train_dataset = MultiConfigDataset(
            data_config_paths=data_config_paths,
            split='train',
            augmentation_config=supcon_config['supcon']['augmentation']
        )
        num_classes = len(train_dataset.categories)
        similarity_matrix = train_dataset.get_similarity_matrix()
        default_similarity = train_dataset.default_similarity
        logger.info(f"合并了 {len(data_config_paths)} 个数据配置文件")
        for i, config_path in enumerate(data_config_paths):
            logger.info(f"  配置 {i+1}: {config_path} ({len(train_dataset.datasets[i])} 个样本)")
    
    logger.info(f"训练集样本数: {len(train_dataset)}, 类别数: {num_classes}")
    logger.info(f"默认相似度阈值: {default_similarity}")
    
    # 创建数据加载器
    batch_size = supcon_config['supcon']['data']['batch_size']
    
    # 使用普通shuffle
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=supcon_config['supcon']['data']['num_workers'],
        pin_memory=supcon_config['supcon']['data']['pin_memory']
    )
    
    # 创建验证集（如果启用验证）
    val_dataset = None
    val_dataloader = None
    if args.use_eval:
        if len(data_config_paths) == 1:
            val_dataset = SupConDataset(
                data_config_path=data_config_paths[0],
                split='val',
                augmentation_config=None  # 验证集不使用增强
            )
        else:
            val_dataset = MultiConfigDataset(
                data_config_paths=data_config_paths,
                split='val',
                augmentation_config=None  # 验证集不使用增强
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=supcon_config['supcon']['data']['num_workers'],
            pin_memory=supcon_config['supcon']['data']['pin_memory']
        )
        logger.info(f"验证集样本数: {len(val_dataset)}")
    
    # 创建模型
    logger.info("创建模型...")
    model_config = supcon_config['supcon']['model']
    model = SupConModel(
        model_name=model_config.get('model_name', 'facebook/dinov3-convnext-small-pretrain-lvd1689m'),
        embedding_dim=model_config['embedding_dim'],
        projection_hidden_dims=model_config['projection_head']['hidden_dims'],
        image_size=224,
        freeze_backbone=supcon_config['supcon']['training_strategy'].get('freeze_backbone_epochs', 0) > 0,
        use_layers=model_config.get('use_layers', None),
        fusion_dim=model_config.get('fusion_dim', 512)
    ).to(device)
    
    
    # 创建Loss（传入相似度矩阵和默认相似度）
    loss_config = supcon_config['supcon']['loss']
    loss_type = loss_config.get('type', 'contrastive')  # 'contrastive' 或 'target'
    
    if loss_type == 'target':
        # 使用MSE loss，相似度作为目标值
        logger.info("使用SimilarityTargetLoss（相似度作为目标值）")
        criterion = SimilarityTargetLoss(
            similarity_matrix=similarity_matrix,
            default_similarity=default_similarity
        ).to(device)
    else:
        # 使用标准对比loss，相似度作为权重
        logger.info("使用SupervisedContrastiveLoss（相似度作为权重）")
        criterion = SupervisedContrastiveLoss(
            temperature=loss_config['supcon']['temperature'],
            similarity_matrix=similarity_matrix,
            default_similarity=default_similarity
        ).to(device)
    
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
    best_margin = float('inf')
    if args.resume:
        logger.info(f"从checkpoint恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_margin = checkpoint.get('best_margin', float('inf'))
    
    # 训练循环
    logger.info("开始训练...")
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, training_config['epochs']+1):
        # 解冻backbone（如果需要）
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            logger.info("解冻backbone参数...")
            model.unfreeze_all()
        
        # 训练
        train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch,
        )
        train_losses.append(train_metrics['loss'])
        
        # 验证
        val_metrics = None
        if args.use_eval and val_dataloader is not None:
            val_metrics = validate(model, val_dataloader, criterion, device)
            val_losses.append(val_metrics['loss'])
        
        # 更新学习率
        scheduler.step()
         
        # 记录日志（使用相似度分布分析指标）
        log_msg = (
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )
        if val_metrics is not None:
            log_msg += (
                f", val_loss={val_metrics['loss']:.4f}\n"
                f"  Margin Stats: "
                f"PosSim={val_metrics.get('margin_pos_sim', 0.0):.4f}, "
                f"NegSim={val_metrics.get('margin_neg_sim', 0.0):.4f}, "
                f"Margin={val_metrics.get('margin', 0.0):.4f}\n"
                f"  kNN Accuracy: {val_metrics.get('knn_accuracy', 0.0):.4f}"
            )
        logger.info(log_msg)
        
        # Wandb记录（使用相似度分布分析指标）
        if supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'learning_rate': scheduler.get_last_lr()[0]
            }
            if val_metrics is not None:
                log_dict.update({
                    'val_loss': val_metrics['loss'],
                    # 相似度分布统计（Margin指标）
                    'val_margin_pos_sim': val_metrics.get('margin_pos_sim', 0.0),
                    'val_margin_neg_sim': val_metrics.get('margin_neg_sim', 0.0),
                    'val_margin': val_metrics.get('margin', 0.0),
                    # kNN评估指标（独立指标）
                    'val_knn_accuracy': val_metrics.get('knn_accuracy', 0.0),
                })
            wandb.log(log_dict)
        
        # 保存checkpoint
        if epoch % training_config.get('save_interval', 5) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_metrics['loss'],
                'best_margin': best_margin,
                'config': supcon_config
            }
            if val_metrics is not None:
                checkpoint['val_loss'] = val_metrics['loss']
            torch.save(
                checkpoint,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            )

        # 保存当前模型（覆盖）
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_metrics['loss'],
                'best_margin': best_margin,
                'config': supcon_config
            },
            checkpoint_dir / "current_model.pth"
        )
        
    # 绘制损失曲线
    plot_loss_curve(
        train_losses,
        val_losses if args.use_eval else [],
        save_path=str(checkpoint_dir / "loss_curve.png"),
        title="SupCon Training Loss"
    )
    
    logger.info("训练完成！")
    if supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
