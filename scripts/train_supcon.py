"""
SupCon监督对比学习训练脚本
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
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None
    
# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.supcon_train.datasets.supcon_dataset import (MultiScaleBatchSampler, SupConDataset,
                                                      multi_scale_collate_fn)
from src.supcon_train.models.losses import ComprehensiveSegmentationLoss, SupervisedContrastiveLoss
from src.supcon_train.models.moco_loss import MoCoLoss
from src.supcon_train.models.moco_model import MoCoModel
from src.supcon_train.models.moco_queue import MoCoQueue
from src.supcon_train.models.supcon_model import SupConModel
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.metrics import knn_evaluation, similarity_distribution_stats
from src.utils.visualization import plot_loss_curve

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class LabelMappingDataset(Dataset):
    """包装数据集，将场景内 label 下标映射为全局 label 下标（供多场景训练 Loss 使用）"""

    def __init__(self, inner_dataset: Dataset, label_mapping: dict):
        self.inner_dataset = inner_dataset
        self.label_mapping = label_mapping  # 场景内 idx -> 全局 idx

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, idx):
        out = self.inner_dataset.__getitem__(idx)
        out = dict(out)
        out['label'] = self.label_mapping[out['label']]
        return out


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_moco: bool = False,
    moco_queue: Optional[MoCoQueue] = None,
    seg_criterion: Optional[nn.Module] = None,
    seg_loss_weight: float = 0.5,
) -> dict:
    """
    训练一个epoch
    
    Args:
        model: 模型（SupConModel或MoCoModel）
        dataloader: 数据加载器
        criterion: 对比损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        use_moco: 是否使用MoCo
        moco_queue: MoCo队列（如果使用MoCo）
        seg_criterion: 分割损失函数（可选）
        seg_loss_weight: 分割损失权重
    """
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_seg_loss = 0.0
    total_pos_loss = 0.0
    total_neg_loss = 0.0
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
        
        if use_moco:
            # MoCo训练流程
            # 1. 使用query_encoder计算view1的query embeddings
            query_outputs1 = model(view1_images, view1_masks, mode='query', return_features=False, return_segmentation=(seg_criterion is not None))
            query_embeddings1 = query_outputs1['embeddings']
            
            # 2. 使用momentum_encoder计算view2的key embeddings（用于positive pairs和更新队列）
            with torch.no_grad():
                key_outputs2 = model(view2_images, view2_masks, mode='key', return_features=False, return_segmentation=False)
                key_embeddings2 = key_outputs2['embeddings']
            
            # 3. 获取队列中的负样本
            queue_embeddings, queue_labels = moco_queue.get_queue(device=device)
            
            # 4. 检查embeddings是否包含NaN或Inf
            if (torch.isnan(query_embeddings1).any() or torch.isinf(query_embeddings1).any() or
                torch.isnan(key_embeddings2).any() or torch.isinf(key_embeddings2).any()):
                nan_embedding_count += 1
                skipped_batches += 1
                if nan_embedding_count <= 5:
                    print(f"警告：Epoch {epoch}, Batch {num_batches}: embeddings包含NaN或Inf，跳过此batch")
                continue
            
            # 5. 计算MoCo loss（结合当前batch和队列中的负样本）
            contrastive_loss = criterion(
                query_embeddings=query_embeddings1,
                key_embeddings=key_embeddings2,
                query_labels=labels,
                queue_embeddings=queue_embeddings,
                queue_labels=queue_labels
            )
            
            # 计算分割损失（如果启用）
            seg_loss = torch.tensor(0.0, device=device)
            if seg_criterion is not None and 'segmentation' in query_outputs1:
                seg_pred = query_outputs1['segmentation']  # (B, 1, H, W)
                seg_loss = seg_criterion(seg_pred, view1_masks)
            
            # 总损失
            loss = contrastive_loss + seg_loss_weight * seg_loss
            
            # 6. 检查loss是否为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
                nan_loss_count += 1
                skipped_batches += 1
                if nan_loss_count <= 5:
                    print(f"警告：Epoch {epoch}, Batch {num_batches}: loss为NaN或Inf，跳过此batch")
                continue
            
            # 7. 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.query_encoder.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            # 8. 更新队列（FIFO）
            with torch.no_grad():
                moco_queue.enqueue(key_embeddings2, labels)
            
            # 9. 动量更新momentum_encoder
            with torch.no_grad():
                model.momentum_update()
        else:
            # 标准SupCon训练流程
            # 前向传播：分别计算view1和view2的embedding
            outputs1 = model(view1_images, view1_masks, return_features=False, return_segmentation=(seg_criterion is not None))
            embeddings1 = outputs1['embeddings']
            
            outputs2 = model(view2_images, view2_masks, return_features=False, return_segmentation=False)
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

            contrastive_loss = criterion(embeddings, labels_duplicated)
            
            # 计算分割损失（如果启用）
            seg_loss = torch.tensor(0.0, device=device)
            if seg_criterion is not None and 'segmentation' in outputs1:
                seg_pred = outputs1['segmentation']  # (B, 1, H, W)
                seg_loss = seg_criterion(seg_pred, view1_masks)
            
            # 总损失
            loss = contrastive_loss + seg_loss_weight * seg_loss
            
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
        
        total_loss += loss.item()
        if use_moco:
            total_contrastive_loss += contrastive_loss.item()
        else:
            total_contrastive_loss += contrastive_loss.item()
        if seg_criterion is not None:
            total_seg_loss += seg_loss.item()
        
        # 收集pos_loss和neg_loss（仅MoCo）
        if use_moco and hasattr(criterion, 'last_pos_loss') and criterion.last_pos_loss is not None:
            total_pos_loss += criterion.last_pos_loss
        if use_moco and hasattr(criterion, 'last_neg_loss') and criterion.last_neg_loss is not None:
            total_neg_loss += criterion.last_neg_loss
        
        num_batches += 1
        
        postfix_dict = {'loss': loss.item()}
        if seg_criterion is not None:
            postfix_dict['seg_loss'] = seg_loss.item()
        pbar.set_postfix(postfix_dict)
    
    # 打印统计信息
    if skipped_batches > 0:
        print(f"Epoch {epoch}统计: 跳过{skipped_batches}个batch (NaN embedding: {nan_embedding_count}, NaN loss: {nan_loss_count})")
    
    result = {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'contrastive_loss': total_contrastive_loss / num_batches if num_batches > 0 else 0.0,
        'skipped_batches': skipped_batches,
    }
    
    # 添加分割损失信息
    if seg_criterion is not None and num_batches > 0:
        result['seg_loss'] = total_seg_loss / num_batches
        if hasattr(seg_criterion, 'last_loss_details') and seg_criterion.last_loss_details:
            details = seg_criterion.last_loss_details
            result['seg_pred_ratio'] = details['pred_foreground_ratio']
            result['seg_gt_ratio'] = details['gt_foreground_ratio']
    
    # 添加pos_loss和neg_loss（仅MoCo）
    if use_moco:
        if num_batches > 0:
            result['pos_loss'] = total_pos_loss / num_batches
            result['neg_loss'] = total_neg_loss / num_batches if total_neg_loss > 0 else 0.0
        else:
            result['pos_loss'] = 0.0
            result['neg_loss'] = 0.0
    
    return result


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_moco: bool = False,
    similarity_matrix: Optional[np.ndarray] = None,
    default_similarity: float = 0.0,
    loss_temperature: float = 0.07,
) -> dict:
    """
    验证函数，使用相似度分布分析作为评估指标
    
    评估指标（基于整个验证集计算，而非batch平均）：
    - PosSim: 正样本平均相似度（越大越好，通常趋近1）
    - NegSim: 负样本平均相似度（越小越好，接近0或负值）
    - Margin: 正负样本间隔（越大越好，表示判别性越强）
    
    Args:
        model: 模型（SupConModel或MoCoModel）
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        use_moco: 是否使用MoCo（如果使用MoCo，验证时只使用query_encoder）
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
            if use_moco:
                # MoCo验证：只使用query_encoder
                outputs1 = model(view1_images, view1_masks, mode='query', return_features=False)
            else:
                # 标准SupCon验证
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
            
            # 对于MoCo，验证时使用标准SupCon loss（因为不需要队列）
            # 对于标准SupCon，使用原有的loss
            if use_moco:
                # MoCo验证时，使用标准SupCon loss（需要从criterion中提取相似度矩阵信息）
                # 创建一个临时的SupervisedContrastiveLoss用于验证
                temp_criterion = SupervisedContrastiveLoss(
                    temperature=loss_temperature,
                    similarity_matrix=similarity_matrix,
                    default_similarity=default_similarity
                ).to(device)
                loss = temp_criterion(embeddings1, labels)
            else:
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


def build_model(
    model_config: dict,
    moco_config: dict,
    use_moco: bool,
    image_size: int,
    freeze_backbone: bool,
    device: torch.device,
    logger
) -> tuple:
    """
    构建模型
    
    Args:
        model_config: 模型配置
        moco_config: MoCo配置
        use_moco: 是否使用MoCo
        image_size: 图像大小
        freeze_backbone: 是否冻结backbone
        device: 设备
        logger: 日志记录器
        
    Returns:
        (model, moco_queue): 模型和MoCo队列（如果不使用MoCo则为None）
    """
    if use_moco:
        logger.info("使用MoCo模型（动量对比学习）")
        momentum = moco_config.get('momentum', 0.999)
        logger.info(f"MoCo动量系数: {momentum}")
        # 获取分割相关配置
        seg_config = model_config.get('segmentation', {})
        enable_segmentation = seg_config.get('enabled', True)
        seg_layer_idx = seg_config.get('layer_idx', 0)
        
        if enable_segmentation:
            logger.info(f"启用语义分割分支，使用FPN第{seg_layer_idx}层特征")
        
        model = MoCoModel(
            model_name=model_config.get('model_name', 'facebook/dinov3-convnext-small-pretrain-lvd1689m'),
            embedding_dim=model_config['embedding_dim'],
            projection_hidden_dims=model_config['projection_head']['hidden_dims'],
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            use_layers=model_config.get('use_layers', None),
            fpn_out_channels=model_config.get('fpn_out_channels', 256),
            fusion_dim=model_config.get('fusion_dim', 512),
            momentum=momentum,
            enable_segmentation=enable_segmentation,
            seg_layer_idx=seg_layer_idx
        ).to(device)
        
        # 创建MoCo队列
        queue_size = moco_config.get('queue_size', 16384)
        logger.info(f"MoCo队列大小: {queue_size}")
        moco_queue = MoCoQueue(
            queue_size=queue_size,
            embedding_dim=model_config['embedding_dim']
        ).to(device)
    else:
        logger.info("使用标准SupCon模型")
        # 获取分割相关配置
        seg_config = model_config.get('segmentation', {})
        enable_segmentation = seg_config.get('enabled', True)
        seg_layer_idx = seg_config.get('layer_idx', 0)
        
        if enable_segmentation:
            logger.info(f"启用语义分割分支，使用FPN第{seg_layer_idx}层特征")
        
        model = SupConModel(
            model_name=model_config.get('model_name', 'facebook/dinov3-convnext-small-pretrain-lvd1689m'),
            embedding_dim=model_config['embedding_dim'],
            projection_hidden_dims=model_config['projection_head']['hidden_dims'],
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            use_layers=model_config.get('use_layers', None),
            fpn_out_channels=model_config.get('fpn_out_channels', 256),
            fusion_dim=model_config.get('fusion_dim', 512),
            enable_segmentation=enable_segmentation,
            seg_layer_idx=seg_layer_idx
        ).to(device)
        moco_queue = None
    
    return model, moco_queue


def build_loss_func(
    loss_config: dict,
    moco_config: dict,
    use_moco: bool,
    similarity_matrix: Optional[np.ndarray],
    default_similarity: float,
    device: torch.device,
    logger,
    enable_segmentation: bool = True
) -> tuple:
    """
    构建损失函数
    
    Args:
        loss_config: 损失函数配置
        moco_config: MoCo配置
        use_moco: 是否使用MoCo
        similarity_matrix: 相似度矩阵
        default_similarity: 默认相似度
        device: 设备
        logger: 日志记录器
        enable_segmentation: 是否启用分割分支
        
    Returns:
        (对比损失函数, 分割损失函数) 或 (对比损失函数, None)
    """
    # 获取是否使用相似度矩阵的配置
    use_similarity_matrix = loss_config.get('use_similarity_matrix', True)
    
    if use_moco:
        # MoCo Loss
        moco_loss_type = moco_config.get('loss_type', 'supervised')  # 'supervised' 或 'standard'
        logger.info(f"使用MoCoLoss（类型: {moco_loss_type}）")
        
        if moco_loss_type == 'supervised':
            # 监督对比loss（支持相似度矩阵）
            if use_similarity_matrix:
                logger.info("  使用相似度矩阵")
            else:
                logger.info("  不使用相似度矩阵（标准SupCon模式）")
            # 获取负样本惩罚参数（从loss_config中读取，如果不存在则使用默认值）
            neg_weight = loss_config.get('neg_weight', 0.5)
            margin = loss_config.get('margin', 0.0)
            logger.info(f"  负样本惩罚权重: {neg_weight}, margin: {margin}")
            criterion = MoCoLoss(
                temperature=loss_config['supcon']['temperature'],
                loss_type='supervised',
                similarity_matrix=similarity_matrix if use_similarity_matrix else None,
                default_similarity=default_similarity if use_similarity_matrix else 0.0,
                use_similarity_matrix=use_similarity_matrix,
                neg_weight=neg_weight,
                margin=margin
            ).to(device)
        else:
            # 标准MoCo loss（InfoNCE），不使用相似度矩阵
            # 获取负样本惩罚参数（从loss_config中读取，如果不存在则使用默认值）
            neg_weight = loss_config.get('neg_weight', 0.5)
            margin = loss_config.get('margin', 0.0)
            logger.info(f"  负样本惩罚权重: {neg_weight}, margin: {margin}")
            criterion = MoCoLoss(
                temperature=loss_config['supcon']['temperature'],
                loss_type='standard',
                use_similarity_matrix=False,
                neg_weight=neg_weight,
                margin=margin
            ).to(device)
    else:
        # 标准SupCon Loss
        if use_similarity_matrix:
            logger.info("使用SupervisedContrastiveLoss（相似度作为权重，使用相似度矩阵）")
        else:
            logger.info("使用SupervisedContrastiveLoss（标准SupCon模式，仅同label为positive）")
        criterion = SupervisedContrastiveLoss(
            temperature=loss_config['supcon']['temperature'],
            similarity_matrix=similarity_matrix if use_similarity_matrix else None,
            default_similarity=default_similarity if use_similarity_matrix else 0.0,
            use_similarity_matrix=use_similarity_matrix
        ).to(device)
    
    # 构建分割损失函数
    seg_criterion = None
    if enable_segmentation:
        seg_config = loss_config.get('segmentation', {})
        logger.info("构建分割损失函数（ComprehensiveSegmentationLoss）")
        logger.info(f"  BCE权重: {seg_config.get('bce_weight', 0.4)}, "
                   f"Dice权重: {seg_config.get('dice_weight', 0.4)}, "
                   f"比例约束权重: {seg_config.get('ratio_weight', 0.1)}")
        seg_criterion = ComprehensiveSegmentationLoss(
            bce_weight=seg_config.get('bce_weight', 0.4),
            dice_weight=seg_config.get('dice_weight', 0.4),
            ratio_weight=seg_config.get('ratio_weight', 0.1),
            target_foreground_ratio=seg_config.get('target_foreground_ratio', 0.1),
            max_foreground_ratio=seg_config.get('max_foreground_ratio', 0.1)
        ).to(device)
    
    return criterion, seg_criterion


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
    
    # 创建数据集（支持单配置文件内多场景：scenes 列表；无 scenes 时整文件视为单场景）
    logger.info("加载数据集...")
    data_config_path = data_config_paths[0] if data_config_paths else 'configs/data_config_zhenyu.yaml'
    logger.info(f"数据配置文件: {data_config_path}")
    data_config = load_config(data_config_path)

    # 解析场景列表：有 scenes 则为多场景，否则整份配置为一个场景
    if 'scenes' in data_config:
        scene_configs = data_config['scenes']
        scene_names = [s.get('name', f'scene_{i}') for i, s in enumerate(scene_configs)]
        logger.info(f"多场景模式: {len(scene_configs)} 个场景 -> {scene_names}")
    else:
        scene_configs = [data_config]
        scene_names = ['default']

    # 处理image_size配置（可能是单个整数或整数列表）
    image_size_config = supcon_config['supcon']['data'].get('image_size', 224)
    if isinstance(image_size_config, int):
        image_sizes = [image_size_config]
        use_multiscale = False
    elif isinstance(image_size_config, list):
        image_sizes = image_size_config
        use_multiscale = len(image_sizes) > 1
    else:
        raise ValueError(f"image_size必须是int或List[int]，当前为{type(image_size_config)}")

    model_image_size = max(image_sizes)
    val_image_size = max(image_sizes)
    logger.info(f"图像尺度配置: {image_sizes}")
    if use_multiscale:
        logger.info(f"启用多尺度训练: 训练时每个batch随机选择 {image_sizes} 中的一个尺度")
        logger.info(f"验证集固定使用尺度: {val_image_size}")
    logger.info(f"模型初始化使用尺度: {model_image_size}")

    # 为每个场景创建 SupConDataset（不混合）
    train_datasets_raw = []
    val_datasets_raw = []
    for scene_cfg in scene_configs:
        train_ds = SupConDataset(
            data_config=scene_cfg,
            split='train',
            image_size=image_sizes,
        )
        val_ds = SupConDataset(
            data_config=scene_cfg,
            split='val',
            image_size=val_image_size,
        )
        train_datasets_raw.append(train_ds)
        val_datasets_raw.append(val_ds)

    # 全局类别并集与每场景 label 映射
    all_categories = []
    for ds in train_datasets_raw:
        for c in ds.categories:
            if c not in all_categories:
                all_categories.append(c)
    all_categories = sorted(all_categories)
    cat2idx_global = {c: i for i, c in enumerate(all_categories)}
    label_mappings = []
    for ds in train_datasets_raw:
        mapping = {ds.cat2idx[c]: cat2idx_global[c] for c in ds.categories}
        label_mappings.append(mapping)

    # 全局相似度矩阵（合并各场景 custom_similarity / default_similarity）
    all_custom_similarity = []
    for ds in train_datasets_raw:
        all_custom_similarity.extend(ds.custom_similarity)
    default_similarity = train_datasets_raw[0].default_similarity if train_datasets_raw else 0.0
    num_classes = len(all_categories)
    similarity_matrix = np.full((num_classes, num_classes), default_similarity, dtype=np.float64)
    np.fill_diagonal(similarity_matrix, 1.0)
    for item in all_custom_similarity:
        categories_list = item.get('list', [])
        sim = item.get('similarity', default_similarity)
        for i, cat1 in enumerate(categories_list):
            for cat2 in categories_list[i + 1:]:
                if cat1 in cat2idx_global and cat2 in cat2idx_global:
                    idx1, idx2 = cat2idx_global[cat1], cat2idx_global[cat2]
                    similarity_matrix[idx1, idx2] = sim
                    similarity_matrix[idx2, idx1] = sim

    logger.info(f"全局类别数: {num_classes}, 缺陷类别: {all_categories}")
    for i, name in enumerate(scene_names):
        logger.info(f"  场景 {name}: 训练 {len(train_datasets_raw[i])} 样本, 验证 {len(val_datasets_raw[i])} 样本")

    # 包装为全局 label 映射后的 dataset（供 Loss 使用统一类别空间）
    train_datasets = [LabelMappingDataset(ds, label_mappings[i]) for i, ds in enumerate(train_datasets_raw)]
    val_datasets = [LabelMappingDataset(ds, label_mappings[i]) for i, ds in enumerate(val_datasets_raw)]

    # 创建每个场景的 DataLoader 列表
    batch_size = supcon_config['supcon']['data']['batch_size']
    num_workers = supcon_config['supcon']['data']['num_workers']
    pin_memory = supcon_config['supcon']['data']['pin_memory']
    train_dataloaders = []
    for ds in train_datasets:
        if use_multiscale:
            batch_sampler = MultiScaleBatchSampler(
                dataset=ds,
                batch_size=batch_size,
                image_sizes=image_sizes,
                shuffle=True,
            )
            train_dataloaders.append(DataLoader(
                ds,
                batch_sampler=batch_sampler,
                collate_fn=multi_scale_collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ))
        else:
            train_dataloaders.append(DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ))

    val_dataloaders = []
    if args.use_eval:
        for ds in val_datasets:
            val_dataloaders.append(DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ))
        total_val = sum(len(d) for d in val_datasets)
        logger.info(f"验证集总样本数: {total_val}（{len(val_dataloaders)} 个场景）")
    
    # 检查是否使用MoCo
    moco_config = supcon_config['supcon'].get('moco', {})
    use_moco = moco_config.get('enabled', False)
    
    # 创建模型
    logger.info("创建模型...")
    model_config = supcon_config['supcon']['model']
    freeze_backbone = supcon_config['supcon']['training_strategy'].get('freeze_backbone_epochs', 0) > 0
    
    model, moco_queue = build_model(
        model_config=model_config,
        moco_config=moco_config,
        use_moco=use_moco,
        image_size=model_image_size,  # 使用最大尺度初始化模型
        freeze_backbone=freeze_backbone,
        device=device,
        logger=logger
    )

    # 多场景时每个场景使用独立的 MoCo 队列，避免跨场景负样本混合；单场景沿用单个 queue
    if use_moco and moco_queue is not None:
        if len(scene_names) > 1:
            queue_size = moco_config.get('queue_size', 16384)
            emb_dim = model_config['embedding_dim']
            moco_queues = [MoCoQueue(queue_size=queue_size, embedding_dim=emb_dim).to(device) for _ in scene_names]
            logger.info(f"多场景 MoCo: 为 {len(scene_names)} 个场景各创建独立队列 (size={queue_size})")
        else:
            moco_queues = [moco_queue]
    else:
        moco_queues = None

    # 创建Loss（传入相似度矩阵和默认相似度）
    loss_config = supcon_config['supcon']['loss']
    enable_segmentation = model_config.get('segmentation', {}).get('enabled', True)
    criterion, seg_criterion = build_loss_func(
        loss_config=loss_config,
        moco_config=moco_config,
        use_moco=use_moco,
        similarity_matrix=similarity_matrix,
        default_similarity=default_similarity,
        device=device,
        logger=logger,
        enable_segmentation=enable_segmentation
    )
    
    # 获取分割损失权重
    seg_loss_weight = loss_config.get('segmentation', {}).get('weight', 0.5)
    if enable_segmentation:
        logger.info(f"分割损失权重: {seg_loss_weight}")
    
    # 创建优化器（backbone和projection head使用不同学习率）
    training_config = supcon_config['supcon']['training']
    backbone_lr_ratio = training_config.get('backbone_lr_ratio', 0.1)
    
    # 分离backbone和projection head的参数
    backbone_params = []
    other_params = []
    if use_moco:
        # MoCo模型：只优化query_encoder的参数
        for name, param in model.query_encoder.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
    else:
        # 标准SupCon模型
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
        
        # 如果使用MoCo，恢复队列状态（如果checkpoint中有）；支持单队列或 per-scene 队列列表
        if use_moco and moco_queues is not None and 'moco_queue_state' in checkpoint:
            st = checkpoint['moco_queue_state']
            if isinstance(st, list):
                for i, q in enumerate(moco_queues):
                    if i < len(st):
                        q.load_state_dict(st[i])
                logger.info("MoCo队列状态已恢复（多场景）")
            else:
                moco_queues[0].load_state_dict(st)
                logger.info("MoCo队列状态已恢复（单队列）")
    
    # 训练循环
    logger.info("开始训练...")
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, training_config['epochs']+1):
        # 解冻backbone（如果需要）
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            logger.info("解冻backbone参数...")
            model.unfreeze_all()

        # 每 epoch 依次训练所有场景（每个场景使用自己的 MoCo 队列）
        train_metrics_per_scene = []
        for scene_idx, (scene_name, train_dl) in enumerate(zip(scene_names, train_dataloaders)):
            scene_queue = moco_queues[scene_idx] if moco_queues is not None else None
            metrics = train_epoch(
                model, train_dl, criterion, optimizer, device, epoch,
                use_moco=use_moco,
                moco_queue=scene_queue,
                seg_criterion=seg_criterion,
                seg_loss_weight=seg_loss_weight,
            )
            train_metrics_per_scene.append((scene_name, metrics))
        train_metrics = {
            'loss': sum(m['loss'] for _, m in train_metrics_per_scene) / len(train_metrics_per_scene),
            'contrastive_loss': sum(m.get('contrastive_loss', 0) for _, m in train_metrics_per_scene) / len(train_metrics_per_scene),
            'skipped_batches': sum(m.get('skipped_batches', 0) for _, m in train_metrics_per_scene),
        }
        if train_metrics_per_scene[0][1].get('seg_loss') is not None:
            train_metrics['seg_loss'] = sum(m.get('seg_loss', 0) for _, m in train_metrics_per_scene) / len(train_metrics_per_scene)
        if use_moco and train_metrics_per_scene[0][1].get('pos_loss') is not None:
            train_metrics['pos_loss'] = sum(m.get('pos_loss', 0) for _, m in train_metrics_per_scene) / len(train_metrics_per_scene)
            train_metrics['neg_loss'] = sum(m.get('neg_loss', 0) for _, m in train_metrics_per_scene) / len(train_metrics_per_scene)
        train_losses.append(train_metrics['loss'])

        # 验证：所有场景单独计算指标
        val_metrics = None
        val_metrics_per_scene = []
        if args.use_eval and val_dataloaders:
            for scene_name, val_dl in zip(scene_names, val_dataloaders):
                vm = validate(
                    model, val_dl, criterion, device,
                    use_moco=use_moco,
                    similarity_matrix=similarity_matrix,
                    default_similarity=default_similarity,
                    loss_temperature=loss_config['supcon']['temperature'],
                )
                val_metrics_per_scene.append((scene_name, vm))
            # 整体验证指标取各场景均值（用于 best_margin / 日志汇总）
            val_metrics = {
                'loss': sum(m['loss'] for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                'margin_pos_sim': sum(m.get('margin_pos_sim', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                'margin_neg_sim': sum(m.get('margin_neg_sim', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                'margin': sum(m.get('margin', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                'knn_accuracy': sum(m.get('knn_accuracy', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
            }
            val_losses.append(val_metrics['loss'])

        # 更新学习率
        scheduler.step()

        # 记录日志（含每场景训练/验证）
        log_msg = (
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )
        if use_moco:
            if 'pos_loss' in train_metrics:
                log_msg += f", pos_loss={train_metrics['pos_loss']:.4f}"
            if 'neg_loss' in train_metrics:
                log_msg += f", neg_loss={train_metrics['neg_loss']:.4f}"
        for sn, m in train_metrics_per_scene:
            log_msg += f"\n  [train] {sn}: loss={m['loss']:.4f}"
        if val_metrics is not None:
            log_msg += (
                f"\n  [val] 汇总: loss={val_metrics['loss']:.4f}, "
                f"PosSim={val_metrics.get('margin_pos_sim', 0):.4f}, "
                f"NegSim={val_metrics.get('margin_neg_sim', 0):.4f}, "
                f"Margin={val_metrics.get('margin', 0):.4f}, kNN={val_metrics.get('knn_accuracy', 0):.4f}"
            )
            if moco_queues is not None:
                for sn, q in zip(scene_names, moco_queues):
                    log_msg += f"\n  moco_queue {sn} full: {q.is_full()}"
            for sn, vm in val_metrics_per_scene:
                log_msg += (
                    f"\n  [val] {sn}: loss={vm['loss']:.4f}, "
                    f"Margin={vm.get('margin', 0):.4f}, kNN={vm.get('knn_accuracy', 0):.4f}"
                )
        logger.info(log_msg)

        # Wandb：按场景记录验证指标
        if supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'learning_rate': scheduler.get_last_lr()[0],
            }
            if 'contrastive_loss' in train_metrics:
                log_dict['train_contrastive_loss'] = train_metrics['contrastive_loss']
            if 'seg_loss' in train_metrics:
                log_dict['train_seg_loss'] = train_metrics['seg_loss']
            if 'seg_pred_ratio' in train_metrics_per_scene[0][1]:
                log_dict['train_seg_pred_ratio'] = sum(m.get('seg_pred_ratio', 0) for _, m in train_metrics_per_scene) / len(train_metrics_per_scene)
            if 'seg_gt_ratio' in train_metrics_per_scene[0][1]:
                log_dict['train_seg_gt_ratio'] = sum(m.get('seg_gt_ratio', 0) for _, m in train_metrics_per_scene) / len(train_metrics_per_scene)
            if use_moco:
                if 'pos_loss' in train_metrics:
                    log_dict['train_pos_loss'] = train_metrics['pos_loss']
                if 'neg_loss' in train_metrics:
                    log_dict['train_neg_loss'] = train_metrics['neg_loss']
            if val_metrics is not None:
                log_dict['val_loss'] = val_metrics['loss']
                log_dict['val_margin_pos_sim'] = val_metrics.get('margin_pos_sim', 0)
                log_dict['val_margin_neg_sim'] = val_metrics.get('margin_neg_sim', 0)
                log_dict['val_margin'] = val_metrics.get('margin', 0)
                log_dict['val_knn_accuracy'] = val_metrics.get('knn_accuracy', 0)
                for sn, vm in val_metrics_per_scene:
                    log_dict[f'val_loss/{sn}'] = vm['loss']
                    log_dict[f'val_margin/{sn}'] = vm.get('margin', 0)
                    log_dict[f'val_knn_accuracy/{sn}'] = vm.get('knn_accuracy', 0)
            wandb.log(log_dict)

        # 保存 checkpoint
        if epoch % training_config.get('save_interval', 5) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_metrics['loss'],
                'best_margin': best_margin,
                'config': supcon_config,
            }
            if val_metrics is not None:
                checkpoint['val_loss'] = val_metrics['loss']
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_metrics['loss'],
            'best_margin': best_margin,
            'config': supcon_config,
        }
        if use_moco and moco_queues is not None:
            checkpoint_data['moco_queue_state'] = [q.state_dict() for q in moco_queues]
        torch.save(checkpoint_data, checkpoint_dir / "current_model.pth")

    # 绘制损失曲线
    plot_loss_curve(
        train_losses,
        val_losses if args.use_eval else [],
        save_path=str(checkpoint_dir / "loss_curve.png"),
        title="SupCon Training Loss",
    )
    
    logger.info("训练完成！")
    if supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
