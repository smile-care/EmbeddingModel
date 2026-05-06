"""
SupCon监督对比学习训练脚本（支持多GPU DDP训练）

用法：
  单GPU:  python scripts/train_supcon.py --config ...
  多GPU:  torchrun --nproc_per_node=4 scripts/train_supcon.py --config ...
"""
import argparse
import logging as _logging
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.supcon_train.datasets.supcon_dataset import (MultiScaleBatchSampler, SupConDataset,
                                                      multi_scale_collate_fn)
from src.supcon_train.models.backbone.dinov3_convnext import DINOv3ConvNextConfig
from src.supcon_train.models.backbone.dinov3_vit import DINOv3ViTConfig
from src.supcon_train.models.convnext_model import ConvNeXtModel
from src.supcon_train.models.losses import ComprehensiveSegmentationLoss, SupervisedContrastiveLoss
from src.supcon_train.models.moco_loss import MoCoLoss
from src.supcon_train.models.moco_model import MoCoModel
from src.supcon_train.models.moco_queue import MoCoQueue
from src.supcon_train.models.vit_model import ViTModel
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.metrics import knn_evaluation, similarity_distribution_stats
from src.utils.visualization import plot_loss_curve

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def sanitize_wandb_project_name(project_name: str, default: str = "industrial-supcon") -> str:
    """
    清洗 W&B project 名称，避免非法字符导致初始化失败。

    W&B 不允许以下字符：/, \\, #, ?, %, :
    """
    if not project_name:
        return default

    sanitized = re.sub(r"[\/\\#\?%:]+", "-", str(project_name)).strip()
    sanitized = re.sub(r"\s+", "-", sanitized)
    sanitized = sanitized.strip("-_.")
    return sanitized or default


def parse_queue_size_config(queue_size_config) -> tuple[int, int]:
    """
    统一解析 MoCo queue_size 配置，支持:
    - int: 固定队列大小
    - list/tuple: [min_queue_size, max_queue_size]

    Returns:
        (min_queue_size, max_queue_size)
    """
    if isinstance(queue_size_config, int):
        values = (queue_size_config, queue_size_config)
    elif isinstance(queue_size_config, (list, tuple)):
        if len(queue_size_config) == 1:
            values = (queue_size_config[0], queue_size_config[0])
        elif len(queue_size_config) == 2:
            values = (queue_size_config[0], queue_size_config[1])
        else:
            raise ValueError(f"queue_size 列表长度必须为1或2，当前为 {queue_size_config}")
    else:
        raise ValueError(
            f"queue_size 必须是 int 或 List[int]（如 16384 或 [1024, 16384]），当前类型: {type(queue_size_config)}"
        )

    min_queue_size, max_queue_size = values
    if not isinstance(min_queue_size, int) or not isinstance(max_queue_size, int):
        raise ValueError(f"queue_size 元素必须为整数，当前为 {queue_size_config}")
    if min_queue_size <= 0 or max_queue_size <= 0:
        raise ValueError(f"queue_size 元素必须 > 0，当前为 {queue_size_config}")
    if min_queue_size > max_queue_size:
        raise ValueError(f"queue_size 最小值不能大于最大值，当前为 {queue_size_config}")

    return min_queue_size, max_queue_size


def init_distributed():
    """初始化分布式训练，返回 (local_rank, rank, world_size)。"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    return 0, 0, 1


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
    is_main: bool = True,
) -> dict:
    """
    训练一个epoch

    Args:
        model: 模型（SupConModel或MoCoModel，可能已包装为DDP）
        dataloader: 数据加载器
        criterion: 对比损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        use_moco: 是否使用MoCo
        moco_queue: MoCo队列（如果使用MoCo）
        seg_criterion: 分割损失函数（可选）
        seg_loss_weight: 分割损失权重
        is_main: 是否为主进程（rank 0）
    """
    model.train()
    # 通过 .module 访问 DDP 内部模型的属性/方法
    raw_model = model.module if isinstance(model, DDP) else model

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_seg_loss = 0.0
    total_pos_loss = 0.0
    total_neg_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    nan_embedding_count = 0
    nan_loss_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main)
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

            # 4. 检查并清理 embeddings 中的 NaN/Inf（DDP 下避免提前 continue 导致各 rank 步数不一致）
            if (not torch.isfinite(query_embeddings1).all() or
                not torch.isfinite(key_embeddings2).all()):
                nan_embedding_count += 1
                skipped_batches += 1
                if nan_embedding_count <= 5:
                    print(f"警告：Epoch {epoch}, Batch {num_batches}: embeddings包含NaN或Inf，已自动清理")
                query_embeddings1 = torch.nan_to_num(query_embeddings1, nan=0.0, posinf=1.0, neginf=-1.0)
                key_embeddings2 = torch.nan_to_num(key_embeddings2, nan=0.0, posinf=1.0, neginf=-1.0)

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

            # 6. 检查loss是否为NaN或Inf（DDP 下不提前 continue，改为零损失保持图连通）
            if not torch.isfinite(loss):
                nan_loss_count += 1
                skipped_batches += 1
                if nan_loss_count <= 5:
                    print(f"警告：Epoch {epoch}, Batch {num_batches}: loss为NaN或Inf，使用零损失继续")
                loss = query_embeddings1.sum() * 0.0

            # 7. 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(raw_model.query_encoder.parameters(), max_norm=2.0)

            optimizer.step()

            # 8. 更新队列（FIFO）
            with torch.no_grad():
                moco_queue.enqueue(key_embeddings2, labels)

            # 9. 动量更新momentum_encoder
            with torch.no_grad():
                raw_model.momentum_update()
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

            # 检查并清理 embeddings 中的 NaN/Inf（DDP 下避免提前 continue 导致各 rank 步数不一致）
            if not torch.isfinite(embeddings).all():
                nan_embedding_count += 1
                skipped_batches += 1
                if nan_embedding_count <= 5:  # 只打印前5次警告
                    print(f"警告：Epoch {epoch}, Batch {num_batches}: embeddings包含NaN或Inf，已自动清理")
                embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)


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

            # 检查loss是否为NaN或Inf（DDP 下不提前 continue，改为零损失保持图连通）
            if not torch.isfinite(loss):
                nan_loss_count += 1
                skipped_batches += 1
                if nan_loss_count <= 5:  # 只打印前5次警告
                    print(f"警告：Epoch {epoch}, Batch {num_batches}: loss为NaN或Inf，使用零损失继续")
                loss = embeddings.sum() * 0.0

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
    if skipped_batches > 0 and is_main:
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
    loss_temperature: float = 0.07,
) -> dict:
    """
    验证函数，使用相似度分布分析作为评估指标

    评估指标（基于整个验证集计算，而非batch平均）：
    - PosSim: 正样本平均相似度（越大越好，通常趋近1）
    - NegSim: 负样本平均相似度（越小越好，接近0或负值）
    - Margin: 正负样本间隔（越大越好，表示判别性越强）

    Args:
        model: 模型（未包装的原始模型，不需要是DDP）
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
                # MoCo验证时，使用标准SupCon loss
                # 创建一个临时的SupervisedContrastiveLoss用于验证
                temp_criterion = SupervisedContrastiveLoss(
                    temperature=loss_temperature,
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

            # 收集embeddings and labels（用于全局相似度分布分析）
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
    # 解析 backbone 名称，加载对应架构配置和预训练权重
    import yaml as _yaml
    backbone_name = model_config.get('backbone')
    if not backbone_name:
        raise ValueError("model_config 中必须指定 backbone 字段")
    backbone_cfg_path  = f"configs/backbone/{backbone_name}.yaml"
    backbone_ckpt_path = f"pretrain_ckpts/{backbone_name}.pth"
    logger.info(f"Backbone: {backbone_name}")

    with open(backbone_cfg_path) as _f:
        _raw_cfg = _yaml.safe_load(_f)
    model_type = _raw_cfg.get('model_type', 'dinov3_convnext')
    is_vit     = (model_type == 'dinov3_vit')

    if is_vit:
        backbone_cfg = DINOv3ViTConfig.from_dict(_raw_cfg)
        logger.info("管线: ViT → MaskWeightedPooling → ProjectionHead")
    else:
        backbone_cfg = DINOv3ConvNextConfig.from_dict(_raw_cfg)
        logger.info("管线: ConvNeXt → PA-FPN → FeatureFusion → ProjectionHead")

    # 分割配置
    seg_cfg             = model_config.get('segmentation', {})
    enable_segmentation = seg_cfg.get('enabled', True)
    if enable_segmentation:
        logger.info("启用语义分割辅助分支")

    # 公共参数（两条管线均有效）
    common_kwargs = dict(
        backbone_cfg=backbone_cfg,
        ckpt_path=backbone_ckpt_path,
        embedding_dim=model_config['embedding_dim'],
        projection_hidden_dims=model_config['projection_head']['hidden_dims'],
        image_size=image_size,
        freeze_backbone=freeze_backbone,
        fusion_dim=model_config.get('fusion_dim', 512),
        enable_segmentation=enable_segmentation,
    )

    # 管线专用参数
    if is_vit:
        vit_cfg = model_config.get('vit', {})
        common_kwargs['cls_weight'] = vit_cfg.get('cls_weight', 0.3)
    else:
        cnx_cfg = model_config.get('convnext', {})
        common_kwargs.update(dict(
            use_layers=cnx_cfg.get('use_layers', [0, 1, 2, 3]),
            fpn_out_channels=cnx_cfg.get('fpn_out_channels', 256),
            seg_layer_idx=cnx_cfg.get('seg_layer_idx', 0),
        ))

    if use_moco:
        momentum = moco_config.get('momentum', 0.999)
        logger.info(f"MoCo 动量对比学习，momentum={momentum}")
        model = MoCoModel(**common_kwargs, momentum=momentum).to(device)

        if enable_segmentation and not is_vit:
            logger.info(f"启用语义分割分支，使用FPN第{seg_layer_idx}层特征") # type: ignore

        # 创建MoCo队列
        queue_size_config = moco_config.get('queue_size', 16384)
        _, queue_size = parse_queue_size_config(queue_size_config)
        logger.info(f"MoCo队列大小: {queue_size_config}")
        moco_queue = MoCoQueue(
            queue_size=queue_size,
            embedding_dim=model_config['embedding_dim'],
        ).to(device)
    else:
        model_cls = ViTModel if is_vit else ConvNeXtModel
        logger.info(f"标准 SupCon 模型: {model_cls.__name__}")
        model      = model_cls(**common_kwargs).to(device)
        moco_queue = None

    return model, moco_queue


def build_loss_func(
    loss_config: dict,
    moco_config: dict,
    use_moco: bool,
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
        device: 设备
        logger: 日志记录器
        enable_segmentation: 是否启用分割分支

    Returns:
        (对比损失函数, 分割损失函数) 或 (对比损失函数, None)
    """
    if use_moco:
        # MoCo Loss
        moco_loss_type = moco_config.get('loss_type', 'supervised')  # 'supervised' 或 'standard'
        logger.info(f"使用MoCoLoss（类型: {moco_loss_type}）")

        if moco_loss_type == 'supervised':
            # 监督对比loss
            # 获取负样本惩罚参数（从loss_config中读取，如果不存在则使用默认值）
            neg_weight = loss_config.get('neg_weight', 0.5)
            margin = loss_config.get('margin', 0.0)
            logger.info(f"  负样本惩罚权重: {neg_weight}, margin: {margin}")
            criterion = MoCoLoss(
                temperature=loss_config['supcon']['temperature'],
                loss_type='supervised',
                neg_weight=neg_weight,
                margin=margin
            ).to(device)
        else:
            # 标准MoCo loss（InfoNCE）
            # 获取负样本惩罚参数（从loss_config中读取，如果不存在则使用默认值）
            neg_weight = loss_config.get('neg_weight', 0.5)
            margin = loss_config.get('margin', 0.0)
            logger.info(f"  负样本惩罚权重: {neg_weight}, margin: {margin}")
            criterion = MoCoLoss(
                temperature=loss_config['supcon']['temperature'],
                loss_type='standard',
                neg_weight=neg_weight,
                margin=margin
            ).to(device)
    else:
        # 标准SupCon Loss
        logger.info("使用SupervisedContrastiveLoss（标准SupCon模式，仅同label为positive）")
        criterion = SupervisedContrastiveLoss(
            temperature=loss_config['supcon']['temperature'],
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
    # ------------------------------------------------------------------ #
    # 初始化分布式训练
    # ------------------------------------------------------------------ #
    local_rank, rank, world_size = init_distributed()
    is_main = (rank == 0)
    is_ddp = (world_size > 1)

    parser = argparse.ArgumentParser(description='SupCon监督对比学习训练')
    parser.add_argument('--config', type=str, default='configs/supcon_config.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--use_eval', action='store_true', default=True, help='是否进行验证')
    parser.add_argument('--no_eval', dest='use_eval', action='store_false', help='禁用验证')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    args = parser.parse_args()

    # 加载配置
    supcon_config = load_config(args.config)
    data_config_path = supcon_config['supcon']['data']['data_config_path']

    # 设置日志（仅主进程输出）
    if is_main:
        logger = setup_logger(
            'supcon_train',
            log_dir=supcon_config['supcon']['output']['log_dir']
        )
    else:
        logger = _logging.getLogger('supcon_train_worker')
        logger.addHandler(_logging.NullHandler())
        logger.setLevel(_logging.CRITICAL)

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    logger.info(f"使用设备: {device}，world_size={world_size}")

    # 创建输出目录
    checkpoint_dir = Path(supcon_config['supcon']['output']['checkpoint_dir'])
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 初始化wandb（仅主进程）
    if is_main and supcon_config['supcon']['output'].get('use_wandb', False):
        if wandb is None:
            logger.warning("wandb not installed, skipping wandb logging")
        else:
            raw_wandb_project = supcon_config['supcon']['output'].get('wandb_project', 'industrial-supcon')
            wandb_project = sanitize_wandb_project_name(raw_wandb_project)
            if wandb_project != raw_wandb_project:
                logger.warning(
                    f"W&B project 名称包含非法字符，已自动清洗: '{raw_wandb_project}' -> '{wandb_project}'"
                )
            wandb.init(project=wandb_project, config=supcon_config['supcon'])

    # 创建数据集
    logger.info("加载数据集...")
    logger.info(f"数据配置文件: {data_config_path}")
    data_config = load_config(data_config_path)

    scenes = data_config['scenes']
    train_scene_cfgs = scenes['train']                # 必填
    val_scene_cfgs   = scenes.get('val', [])          # 可选，为空则跳过验证

    # 处理 image_size 配置
    image_size_config = supcon_config['supcon']['data'].get('image_size', 224)
    if isinstance(image_size_config, int):
        image_sizes = [image_size_config]
        use_multiscale = False
    elif isinstance(image_size_config, list):
        image_sizes = image_size_config
        use_multiscale = len(image_sizes) > 1
    else:
        raise ValueError(f"image_size 必须是 int 或 List[int]，当前为 {type(image_size_config)}")

    model_image_size = max(image_sizes)
    val_image_size   = max(image_sizes)
    logger.info(f"图像尺度配置: {image_sizes}")
    if use_multiscale:
        logger.info(f"启用多尺度训练: 训练时每个 batch 随机选择 {image_sizes} 中的一个尺度")
        logger.info(f"验证集固定使用尺度: {val_image_size}")
    logger.info(f"模型初始化使用尺度: {model_image_size}")

    # 读取 mask_dilation 配置
    mask_dilation_config = supcon_config['supcon']['data'].get('mask_dilation', {'enabled': False})
    logger.info(f"Mask软膨胀配置: {mask_dilation_config}")

    # 创建 train datasets
    train_datasets_raw = [
        SupConDataset(root=s['root'], split='train', image_size=image_sizes, name=s.get('name', s['root']), mask_dilation_config=mask_dilation_config)
        for s in train_scene_cfgs
    ]
    train_scene_names = [s.get('name', s['root']) for s in train_scene_cfgs]

    # 创建 val datasets（可选）
    val_datasets_raw = [
        SupConDataset(root=s['root'], split='val', image_size=val_image_size, name=s.get('name', s['root']), mask_dilation_config=mask_dilation_config)
        for s in val_scene_cfgs
    ]
    val_scene_names = [s.get('name', s['root']) for s in val_scene_cfgs]

    # 每个场景类别独立，label idx 在场景内自洽，无需全局映射
    for name, ds in zip(train_scene_names, train_datasets_raw):
        logger.info(f"  [train] {name}: {len(ds)} 样本, 类别: {ds.categories}")
    for name, ds in zip(val_scene_names, val_datasets_raw):
        logger.info(f"  [val]   {name}: {len(ds)} 样本, 类别: {ds.categories}")

    # 创建每个场景的 DataLoader 列表
    # batch_size 为每张卡的 batch 大小，effective total = batch_size * world_size
    batch_size = supcon_config['supcon']['data']['batch_size']

    # 过滤掉样本数不足 batch_size 的子数据集（不足则每卡 0 个 batch，训练无意义）
    min_samples = batch_size * world_size
    filtered = [(n, ds) for n, ds in zip(train_scene_names, train_datasets_raw) if len(ds) >= min_samples]
    dropped = [n for n, ds in zip(train_scene_names, train_datasets_raw) if len(ds) < min_samples]
    if dropped and rank == 0:
        logger.warning(f"以下子数据集样本数 < {min_samples}（batch_size×world_size），已跳过: {dropped}")
        logger.warning(f"共有 {len(filtered)} 个子数据集参与训练，{len(dropped)} 个子数据集被跳过")
    if not filtered:
        raise ValueError(
            f"没有可训练的数据集：所有 train 子数据集样本数都小于 {min_samples}（batch_size×world_size）。"
        )
    train_scene_names = [n for n, _ in filtered]
    train_datasets    = [ds for _, ds in filtered]

    val_datasets = val_datasets_raw
    num_workers = supcon_config['supcon']['data']['num_workers']
    pin_memory = supcon_config['supcon']['data']['pin_memory']
    if rank == 0:
        logger.info(f"batch_size per GPU: {batch_size}, effective total batch_size: {batch_size * world_size} (world_size={world_size})")

    # train_samplers 统一收集所有需要 set_epoch 的采样器（DistributedSampler 或 MultiScaleBatchSampler）
    train_samplers = []
    train_dataloaders = []
    for ds in train_datasets:
        if use_multiscale:
            # 多尺度模式：rank/world_size 直接传入 MultiScaleBatchSampler，
            # 由它负责按 rank 间隔分配 batch，避免 Subset 破坏 IndexWithScale 索引
            batch_sampler = MultiScaleBatchSampler(
                dataset=ds,
                batch_size=batch_size,
                image_sizes=image_sizes,
                shuffle=True,
                drop_last=True,
                rank=rank,
                world_size=world_size,
            )
            train_dataloaders.append(DataLoader(
                ds,
                batch_sampler=batch_sampler,
                collate_fn=multi_scale_collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ))
            train_samplers.append(batch_sampler)  # 需要 set_epoch
        else:
            if is_ddp:
                sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
            else:
                sampler = None
            train_samplers.append(sampler)
            train_dataloaders.append(DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                drop_last=True,
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

    # ------------------------------------------------------------------ #
    # 包装为 DDP（多GPU时）
    # ------------------------------------------------------------------ #
    if is_ddp:
        # 将 BatchNorm 转换为 SyncBatchNorm，避免 DDP 梯度 hook 与 BN inplace 操作冲突
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        logger.info(f"模型已包装为 DDP，GPU 数量: {world_size}")

    # raw_model 用于直接访问模型属性（如 query_encoder、momentum_update、unfreeze_all）
    raw_model = model.module if is_ddp else model

    # 多场景时每个场景使用独立的 MoCo 队列，避免跨场景负样本混合
    # queue_device: "cpu" 表示队列常驻CPU，训练时搬到GPU（显存友好）；"gpu" 表示常驻GPU（速度更快）
    # queue_size 按场景样本数自适应：min(num_samples * 2, config_max)，并对齐到 batch_size
    queue_on_gpu = moco_config.get('queue_device', 'cpu').lower() == 'gpu'
    if use_moco and moco_queue is not None:
        _queue_size_cfg = moco_config.get('queue_size', 16384)
        min_queue_size, max_queue_size = parse_queue_size_config(_queue_size_cfg)
        emb_dim = model_config['embedding_dim']
        if len(train_scene_names) > 1:
            moco_queues = []
            for ds in train_datasets:
                adaptive_size = min(len(ds) * 2, max_queue_size)
                adaptive_size = max(adaptive_size, min_queue_size)
                adaptive_size = max((adaptive_size // batch_size) * batch_size, batch_size)
                q = MoCoQueue(queue_size=adaptive_size, embedding_dim=emb_dim)
                if queue_on_gpu:
                    q = q.to(device)
                moco_queues.append(q)
            for name, q in zip(train_scene_names, moco_queues):
                logger.info(f"  MoCo queue [{name}]: size={q.queue_size}, device={'gpu' if queue_on_gpu else 'cpu'}")
        else:
            moco_queues = [moco_queue.to(device) if queue_on_gpu else moco_queue]
    else:
        moco_queues = None

    # 创建Loss函数
    loss_config = supcon_config['supcon']['loss']
    enable_segmentation = model_config.get('segmentation', {}).get('enabled', True)
    criterion, seg_criterion = build_loss_func(
        loss_config=loss_config,
        moco_config=moco_config,
        use_moco=use_moco,
        device=device,
        logger=logger,
        enable_segmentation=enable_segmentation
    )

    # 获取分割损失权重
    seg_loss_weight = loss_config.get('segmentation', {}).get('weight', 0.5)
    if enable_segmentation:
        logger.info(f"分割损失权重: {seg_loss_weight}")

    # 创建优化器（backbone和projection head使用不同学习率）
    # 使用 raw_model 提取参数，确保与 DDP 内部参数一致
    training_config = supcon_config['supcon']['training']
    backbone_lr_ratio = training_config.get('backbone_lr_ratio', 0.1)

    # 分离backbone和projection head的参数
    backbone_params = []
    other_params = []
    if use_moco:
        # MoCo模型：只优化query_encoder的参数
        for name, param in raw_model.query_encoder.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
    else:
        # 标准SupCon模型
        for name, param in raw_model.named_parameters():
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
        raw_model.load_state_dict(checkpoint['model_state_dict'])
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
        # 通知 DistributedSampler 当前 epoch（保证每个 epoch 乱序不同）
        for sampler in train_samplers:
            if sampler is not None:
                sampler.set_epoch(epoch)

        # 解冻backbone（如果需要）
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            logger.info("解冻backbone参数...")
            raw_model.unfreeze_all()

        # 每 epoch 依次训练所有场景（每个场景使用自己的 MoCo 队列）
        train_metrics_per_scene = []
        for scene_idx, (scene_name, train_dl) in enumerate(zip(train_scene_names, train_dataloaders)):
            logger.info(f"训练场景({scene_idx+1}/{len(train_scene_names)}): {scene_name}")
            scene_queue = None
            if moco_queues is not None:
                if queue_on_gpu:
                    scene_queue = moco_queues[scene_idx]          # 已在 GPU，无需搬运
                else:
                    scene_queue = moco_queues[scene_idx].to(device)  # CPU → GPU
            metrics = train_epoch(
                model, train_dl, criterion, optimizer, device, epoch,
                use_moco=use_moco,
                moco_queue=scene_queue,
                seg_criterion=seg_criterion,
                seg_loss_weight=seg_loss_weight,
                is_main=is_main,
            )
            if scene_queue is not None and not queue_on_gpu:
                moco_queues[scene_idx] = scene_queue.cpu()        # GPU → CPU
            train_metrics_per_scene.append((scene_name, metrics))

        # 多GPU时聚合各进程的训练loss
        if is_ddp:
            for _, metrics in train_metrics_per_scene:
                for key in ('loss', 'contrastive_loss'):
                    if key in metrics:
                        t = torch.tensor(metrics[key], device=device)
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                        metrics[key] = (t / world_size).item()

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

        # 验证：仅在主进程（rank 0）上运行，使用 raw_model 绕过 DDP
        val_metrics = None
        val_metrics_per_scene = []
        if args.use_eval and val_dataloaders and is_main:
            for scene_idx, (scene_name, val_dl) in enumerate(zip(val_scene_names, val_dataloaders)):
                logger.info(f"验证场景({scene_idx+1}/{len(val_scene_names)}): {scene_name}")
                vm = validate(
                    raw_model, val_dl, criterion, device,
                    use_moco=use_moco,
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
                for sn, q in zip(train_scene_names, moco_queues):
                    log_msg += f"\n  moco_queue {sn} full: {q.is_full()}"
            for sn, vm in val_metrics_per_scene:
                log_msg += (
                    f"\n  [val] {sn}: loss={vm['loss']:.4f}, "
                    f"Margin={vm.get('margin', 0):.4f}, kNN={vm.get('knn_accuracy', 0):.4f}"
                )
        logger.info(log_msg)

        # Wandb：按场景记录验证指标（仅主进程）
        if is_main and supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
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

        # 保存 checkpoint（仅主进程）
        if is_main:
            if epoch % training_config.get('save_interval', 5) == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'best_margin': best_margin,
                    'config': supcon_config,
                }
                if val_metrics is not None:
                    checkpoint['val_loss'] = val_metrics['loss']
                if use_moco and moco_queues is not None:
                    checkpoint['moco_queue_state'] = [q.state_dict() for q in moco_queues]
                torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_metrics['loss'],
                'best_margin': best_margin,
                'config': supcon_config,
            }
            if use_moco and moco_queues is not None:
                checkpoint_data['moco_queue_state'] = [q.state_dict() for q in moco_queues]
            torch.save(checkpoint_data, checkpoint_dir / "current_model.pth")

    # 绘制损失曲线（仅主进程）
    if is_main:
        plot_loss_curve(
            train_losses,
            val_losses if args.use_eval else [],
            save_path=str(checkpoint_dir / "loss_curve.png"),
            title="SupCon Training Loss",
        )

    logger.info("训练完成！")
    if is_main and supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
        wandb.finish()

    # 清理分布式进程组
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
