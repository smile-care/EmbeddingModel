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
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None
    
# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.supcon_train.datasets.supcon_dataset import (MultiConfigDataset, MultiScaleBatchSampler,
                                                      SupConDataset, multi_scale_collate_fn)
from src.supcon_train.models.losses import SupervisedContrastiveLoss
from src.supcon_train.models.moco_loss import MoCoLoss
from src.supcon_train.models.moco_model import MoCoModel
from src.supcon_train.models.moco_queue import MoCoQueue
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
    use_moco: bool = False,
    moco_queue: Optional[MoCoQueue] = None,
) -> dict:
    """
    训练一个epoch
    
    Args:
        model: 模型（SupConModel或MoCoModel）
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        use_moco: 是否使用MoCo
        moco_queue: MoCo队列（如果使用MoCo）
    """
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
        
        if use_moco:
            # MoCo训练流程
            # 1. 使用query_encoder计算view1的query embeddings
            query_outputs1 = model(view1_images, view1_masks, mode='query', return_features=False)
            query_embeddings1 = query_outputs1['embeddings']
            
            # 2. 使用momentum_encoder计算view2的key embeddings（用于positive pairs和更新队列）
            with torch.no_grad():
                key_outputs2 = model(view2_images, view2_masks, mode='key', return_features=False)
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
            loss = criterion(
                query_embeddings=query_embeddings1,
                key_embeddings=key_embeddings2,
                query_labels=labels,
                queue_embeddings=queue_embeddings,
                queue_labels=queue_labels
            )
            
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
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
        model = MoCoModel(
            model_name=model_config.get('model_name', 'facebook/dinov3-convnext-small-pretrain-lvd1689m'),
            embedding_dim=model_config['embedding_dim'],
            projection_hidden_dims=model_config['projection_head']['hidden_dims'],
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            use_layers=model_config.get('use_layers', None),
            fpn_out_channels=model_config.get('fpn_out_channels', 256),
            fusion_dim=model_config.get('fusion_dim', 512),
            momentum=momentum
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
        model = SupConModel(
            model_name=model_config.get('model_name', 'facebook/dinov3-convnext-small-pretrain-lvd1689m'),
            embedding_dim=model_config['embedding_dim'],
            projection_hidden_dims=model_config['projection_head']['hidden_dims'],
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            use_layers=model_config.get('use_layers', None),
            fusion_dim=model_config.get('fusion_dim', 512)
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
    logger
) -> nn.Module:
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
        
    Returns:
        损失函数模块
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
            criterion = MoCoLoss(
                temperature=loss_config['supcon']['temperature'],
                loss_type='supervised',
                similarity_matrix=similarity_matrix if use_similarity_matrix else None,
                default_similarity=default_similarity if use_similarity_matrix else 0.0,
                use_similarity_matrix=use_similarity_matrix
            ).to(device)
        else:
            # 标准MoCo loss（InfoNCE），不使用相似度矩阵
            criterion = MoCoLoss(
                temperature=loss_config['supcon']['temperature'],
                loss_type='standard',
                use_similarity_matrix=False
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
    
    return criterion


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
    
    # 确定模型初始化时使用的image_size（使用最大尺度）
    model_image_size = max(image_sizes)
    # 验证集使用的image_size（固定使用最大尺度）
    val_image_size = max(image_sizes)
    
    logger.info(f"图像尺度配置: {image_sizes}")
    if use_multiscale:
        logger.info(f"启用多尺度训练: 训练时每个batch随机选择 {image_sizes} 中的一个尺度")
        logger.info(f"验证集固定使用尺度: {val_image_size}")
    logger.info(f"模型初始化使用尺度: {model_image_size}")
    
    # 如果只有一个配置文件，直接使用 SupConDataset；否则使用 MultiConfigDataset
    if len(data_config_paths) == 1:
        train_dataset = SupConDataset(
            data_config_path=data_config_paths[0],
            split='train',
            image_size=image_sizes  # 传递列表（即使只有一个元素）
        )
        num_classes = len(train_dataset.categories)
        similarity_matrix = train_dataset.get_similarity_matrix()
        default_similarity = train_dataset.default_similarity
    else:
        train_dataset = MultiConfigDataset(
            data_config_paths=data_config_paths,
            split='train',
            image_size=image_sizes  # 传递列表（即使只有一个元素）
        )
        num_classes = len(train_dataset.categories)
        similarity_matrix = train_dataset.get_similarity_matrix()
        default_similarity = train_dataset.default_similarity
        logger.info(f"合并了 {len(data_config_paths)} 个数据配置文件")
        for i, config_path in enumerate(data_config_paths):
            logger.info(f"  配置 {i+1}: {config_path} ({len(train_dataset.datasets[i])} 个样本)")
    
    logger.info(f"训练集样本数: {len(train_dataset)}, 类别数: {num_classes}")
    # logger.info(f"默认相似度阈值: {default_similarity}")
    logger.info(f"缺陷类别: {train_dataset.categories}")
    
    # 创建数据加载器
    batch_size = supcon_config['supcon']['data']['batch_size']
    
    # 根据是否使用多尺度，选择不同的数据加载方式
    if use_multiscale:
        # 使用MultiScaleBatchSampler
        batch_sampler = MultiScaleBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            image_sizes=image_sizes,
            shuffle=True
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=multi_scale_collate_fn,
            num_workers=supcon_config['supcon']['data']['num_workers'],
            pin_memory=supcon_config['supcon']['data']['pin_memory']
        )
    else:
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
                image_size=val_image_size  # 验证集使用固定尺度
            )
        else:
            val_dataset = MultiConfigDataset(
                data_config_paths=data_config_paths,
                split='val',
                image_size=val_image_size  # 验证集使用固定尺度
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=supcon_config['supcon']['data']['num_workers'],
            pin_memory=supcon_config['supcon']['data']['pin_memory']
        )
        logger.info(f"验证集样本数: {len(val_dataset)}")
    
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
    
    # 创建Loss（传入相似度矩阵和默认相似度）
    loss_config = supcon_config['supcon']['loss']
    criterion = build_loss_func(
        loss_config=loss_config,
        moco_config=moco_config,
        use_moco=use_moco,
        similarity_matrix=similarity_matrix,
        default_similarity=default_similarity,
        device=device,
        logger=logger
    )
    
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
        
        # 如果使用MoCo，恢复队列状态（如果checkpoint中有）
        if use_moco and moco_queue is not None and 'moco_queue_state' in checkpoint:
            moco_queue.load_state_dict(checkpoint['moco_queue_state'])
            logger.info("MoCo队列状态已恢复")
    
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
            use_moco=use_moco,
            moco_queue=moco_queue
        )
        train_losses.append(train_metrics['loss'])
        
        # 验证
        val_metrics = None
        if args.use_eval and val_dataloader is not None:
            val_metrics = validate(
                model, val_dataloader, criterion, device,
                use_moco=use_moco,
                similarity_matrix=similarity_matrix,
                default_similarity=default_similarity,
                loss_temperature=loss_config['supcon']['temperature']
            )
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
                f"  kNN Accuracy: {val_metrics.get('knn_accuracy', 0.0):.4f}\n"
                f"  moco_queue is full: {moco_queue.is_full() if moco_queue is not None else 'N/A'}"
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
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_metrics['loss'],
            'best_margin': best_margin,
            'config': supcon_config
        }
        # 如果使用MoCo，保存队列状态
        if use_moco and moco_queue is not None:
            checkpoint_data['moco_queue_state'] = moco_queue.state_dict()
        torch.save(
            checkpoint_data,
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
