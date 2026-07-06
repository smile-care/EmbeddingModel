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
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_model.supcon.datasets.supcon_dataset import (
    MultiSceneSupConDataset,
    SceneBatchSampler,
    SupConDataset,
    multi_scale_collate_fn,
)
from src.embedding_model.supcon.models.backbone.dinov3_convnext import DINOv3ConvNextConfig
from src.embedding_model.supcon.models.backbone.dinov3_vit import DINOv3ViTConfig
from src.embedding_model.supcon.models.convnext_model import ConvNeXtModel
from src.embedding_model.supcon.models.losses import SupervisedContrastiveLoss
from src.embedding_model.supcon.models.moco_loss import MoCoLoss
from src.embedding_model.supcon.models.moco_model import MoCoModel
from src.embedding_model.supcon.models.moco_queue import MoCoQueue
from src.embedding_model.supcon.models.vit_model import ViTModel
from src.embedding_model.utils.config_loader import load_config
from src.embedding_model.utils.logging import setup_logger
from src.embedding_model.utils.metrics import knn_evaluation, similarity_distribution_stats
from src.embedding_model.utils.visualization import plot_loss_curve

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


def train_one_batch(
    model: nn.Module,
    batch: dict,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    global_step: int,
    use_moco: bool = False,
    moco_queue: Optional[MoCoQueue] = None,
) -> dict:
    """训练一个 step；MoCo queue 由调用方按当前 scene 传入。"""
    model.train()
    raw_model = model.module if isinstance(model, DDP) else model

    view1_images = batch['view1_image'].to(device, non_blocking=True)
    view1_masks = batch['view1_mask'].to(device, non_blocking=True)
    view2_images = batch['view2_image'].to(device, non_blocking=True)
    view2_masks = batch['view2_mask'].to(device, non_blocking=True)
    labels = batch['label'].to(device, non_blocking=True)

    skipped_batches = 0
    nan_embedding_count = 0
    nan_loss_count = 0

    if use_moco:
        if moco_queue is None:
            raise ValueError("use_moco=True 时必须传入当前 scene 的 moco_queue")

        query_outputs = model(view1_images, view1_masks, mode='query', return_features=False)
        query_embeddings = query_outputs['projections']
        with torch.no_grad():
            key_outputs = model(view2_images, view2_masks, mode='key', return_features=False)
            key_embeddings = key_outputs['projections']

        queue_embeddings, queue_labels = moco_queue.get_queue(device=device)
        if not torch.isfinite(query_embeddings).all() or not torch.isfinite(key_embeddings).all():
            nan_embedding_count = 1
            skipped_batches = 1
            query_embeddings = torch.nan_to_num(query_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            key_embeddings = torch.nan_to_num(key_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

        contrastive_loss = criterion(
            query_embeddings=query_embeddings,
            key_embeddings=key_embeddings,
            query_labels=labels,
            queue_embeddings=queue_embeddings,
            queue_labels=queue_labels,
        )
        loss = contrastive_loss
        if not torch.isfinite(loss):
            nan_loss_count = 1
            skipped_batches = 1
            loss = query_embeddings.sum() * 0.0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.query_encoder.parameters(), max_norm=2.0)
        optimizer.step()

        with torch.no_grad():
            moco_queue.enqueue(key_embeddings, labels)
            raw_model.momentum_update()

        metrics = {
            'loss': float(loss.item()),
            'contrastive_loss': float(contrastive_loss.item()),
            'skipped_batches': skipped_batches,
            'nan_embedding_count': nan_embedding_count,
            'nan_loss_count': nan_loss_count,
        }
        if hasattr(criterion, 'last_pos_loss') and criterion.last_pos_loss is not None:
            metrics['pos_loss'] = float(criterion.last_pos_loss)
        if hasattr(criterion, 'last_neg_loss') and criterion.last_neg_loss is not None:
            metrics['neg_loss'] = float(criterion.last_neg_loss)
        return metrics

    outputs1 = model(view1_images, view1_masks, return_features=False)
    outputs2 = model(view2_images, view2_masks, return_features=False)
    embeddings = torch.cat([outputs1['projections'], outputs2['projections']], dim=0)
    labels_duplicated = torch.cat([labels, labels], dim=0)

    if not torch.isfinite(embeddings).all():
        nan_embedding_count = 1
        skipped_batches = 1
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

    contrastive_loss = criterion(embeddings, labels_duplicated)
    loss = contrastive_loss
    if not torch.isfinite(loss):
        nan_loss_count = 1
        skipped_batches = 1
        loss = embeddings.sum() * 0.0

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()

    return {
        'loss': float(loss.item()),
        'contrastive_loss': float(contrastive_loss.item()),
        'skipped_batches': skipped_batches,
        'nan_embedding_count': nan_embedding_count,
        'nan_loss_count': nan_loss_count,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_moco: bool = False,
    loss_temperature: float = 0.07,
    embedding_source: str = "representations",
    show_progress: bool = True,
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

    # 收集所有验证集的 projection / configured app embedding 和 labels
    all_projections = []
    all_app_embeddings = []
    all_labels = []
    temp_criterion = None
    if use_moco:
        temp_criterion = SupervisedContrastiveLoss(temperature=loss_temperature).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", disable=not show_progress):
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
            projections1 = outputs1['projections']
            app_embeddings1 = outputs1[embedding_source]

            projections1 = F.normalize(projections1, dim=1, p=2, eps=1e-8)
            app_embeddings1 = F.normalize(app_embeddings1, dim=1, p=2, eps=1e-8)

            # 检查输出是否包含NaN或Inf
            if (torch.isnan(projections1).any() or torch.isinf(projections1).any() or
                torch.isnan(app_embeddings1).any() or torch.isinf(app_embeddings1).any()):
                nan_embedding_count += 1
                skipped_batches += 1
                if nan_embedding_count <= 5:  # 只打印前5次警告
                    print(f"警告：验证时模型输出包含NaN或Inf，跳过此batch")
                continue

            # 对于MoCo，验证时使用标准SupCon loss（因为不需要队列）
            # 对于标准SupCon，使用原有的loss
            if use_moco:
                loss = temp_criterion(projections1, labels)
            else:
                loss = criterion(projections1, labels)

            # 检查loss是否为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
                nan_loss_count += 1
                skipped_batches += 1
                if nan_loss_count <= 5:  # 只打印前5次警告
                    print(f"警告：验证时loss为NaN或Inf，跳过此batch")
                continue

            # 收集 projection / representation and labels
            all_projections.append(projections1.cpu())
            all_app_embeddings.append(app_embeddings1.cpu())
            all_labels.append(labels.cpu())

            total_loss += loss.item()
            num_batches += 1

    # 打印统计信息
    if skipped_batches > 0:
        print(f"验证统计: 跳过{skipped_batches}个batch (NaN embedding: {nan_embedding_count}, NaN loss: {nan_loss_count})")

    # 基于整个验证集计算相似度分布统计（核心评估指标）
    if len(all_projections) > 0:
        # 拼接所有输出和labels
        all_projections_tensor = torch.cat(all_projections, dim=0)  # (N, D_z)
        all_app_embeddings_tensor = torch.cat(all_app_embeddings, dim=0)  # (N, D_app)
        all_labels_tensor = torch.cat(all_labels, dim=0)  # (N,)

        # 将tensor移回device进行计算
        all_projections_tensor = all_projections_tensor.to(device)
        all_app_embeddings_tensor = all_app_embeddings_tensor.to(device)
        all_labels_tensor = all_labels_tensor.to(device)

        # 1. 应用侧 embedding 指标（主指标，来源由配置决定）
        sim_stats = similarity_distribution_stats(all_app_embeddings_tensor, all_labels_tensor)
        margin_pos_sim = sim_stats['pos_sim']
        margin_neg_sim = sim_stats['neg_sim']
        margin = sim_stats['margin']
        knn_stats = knn_evaluation(all_app_embeddings_tensor, all_labels_tensor, k=10)
        knn_accuracy = knn_stats.get('knn_accuracy', 0.0)

        # 2. projection 指标（loss监督空间诊断）
        proj_sim_stats = similarity_distribution_stats(all_projections_tensor, all_labels_tensor)
        projection_margin_pos_sim = proj_sim_stats['pos_sim']
        projection_margin_neg_sim = proj_sim_stats['neg_sim']
        projection_margin = proj_sim_stats['margin']
        projection_knn_stats = knn_evaluation(all_projections_tensor, all_labels_tensor, k=10)
        projection_knn_accuracy = projection_knn_stats.get('knn_accuracy', 0.0)
    else:
        margin_pos_sim = 0.0
        margin_neg_sim = 0.0
        margin = 0.0
        knn_accuracy = 0.0
        projection_margin_pos_sim = 0.0
        projection_margin_neg_sim = 0.0
        projection_margin = 0.0
        projection_knn_accuracy = 0.0

    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        # 相似度分布统计（Margin指标）
        'margin_pos_sim': margin_pos_sim,
        'margin_neg_sim': margin_neg_sim,
        'margin': margin,
        # kNN评估指标（独立指标）
        'knn_accuracy': knn_accuracy,
        'projection_margin_pos_sim': projection_margin_pos_sim,
        'projection_margin_neg_sim': projection_margin_neg_sim,
        'projection_margin': projection_margin,
        'projection_knn_accuracy': projection_knn_accuracy,
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
        logger.info("管线: ConvNeXt → MultiStageMaskPooling → ProjectionHead")

    # 公共参数（两条管线均有效）
    common_kwargs = dict(
        backbone_cfg=backbone_cfg,
        ckpt_path=backbone_ckpt_path,
        embedding_dim=model_config['embedding_dim'],
        image_size=image_size,
        freeze_backbone=freeze_backbone,
    )

    # 管线专用参数
    if is_vit:
        vit_cfg = model_config.get('vit', {})
        common_kwargs['cls_weight'] = vit_cfg.get('cls_weight', 0.3)
        common_kwargs['projection_hidden_dim'] = vit_cfg.get(
            'projection_hidden_dim',
            model_config.get('projection_hidden_dim')
        )
    else:
        cnx_cfg = model_config.get('convnext', {})
        common_kwargs.update(dict(
            use_layers=cnx_cfg.get('use_layers', [1, 2, 3]),
            fusion_dim=cnx_cfg.get('fusion_dim', model_config.get('embedding_dim')),
            projection_hidden_dim=cnx_cfg.get(
                'projection_hidden_dim',
                model_config.get('projection_hidden_dim')
            ),
            mask_gating=cnx_cfg.get('mask_gating', {}),
            pooling=cnx_cfg.get('pooling', {'mode': 'fg_only'}),
        ))

    if use_moco:
        momentum = moco_config.get('momentum', 0.999)
        logger.info(f"MoCo 动量对比学习，momentum={momentum}")
        model = MoCoModel(**common_kwargs, momentum=momentum).to(device)
        moco_queue = None
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
) -> nn.Module:
    """
    构建损失函数

    Args:
        loss_config: 损失函数配置
        moco_config: MoCo配置
        use_moco: 是否使用MoCo
        device: 设备
        logger: 日志记录器
    Returns:
        对比损失函数
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

    return criterion


def get_stage_gate_weights(raw_model: nn.Module, use_moco: bool) -> Optional[dict]:
    """读取 ConvNeXt FeatureFusion 的 stage gate 权重；未启用时返回 None。"""
    encoder = raw_model.query_encoder if use_moco and hasattr(raw_model, "query_encoder") else raw_model
    fusion = getattr(encoder, "feature_fusion", None)
    if fusion is None or not hasattr(fusion, "get_stage_weights"):
        return None

    weights = fusion.get_stage_weights()
    if weights is None:
        return None

    use_layers = getattr(fusion, "use_layers", range(len(weights)))
    return {
        f"stage_{layer_idx}": float(weights[i].cpu().item())
        for i, layer_idx in enumerate(use_layers)
    }


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
    copy_paste_config = supcon_config['supcon']['data'].get('copy_paste', {'enabled': False})
    logger.info(f"Copy-paste干扰增强配置: {copy_paste_config}")

    train_dataset = MultiSceneSupConDataset(
        scene_cfgs=train_scene_cfgs,
        split='train',
        image_size=image_sizes,
        mask_dilation_config=mask_dilation_config,
        copy_paste_config=copy_paste_config,
    )
    train_scene_names = train_dataset.scene_names

    # 创建 val datasets（可选）
    val_datasets_raw = [
        SupConDataset(
            root=s['root'],
            split='val',
            image_size=val_image_size,
            name=s.get('name', s['root']),
            mask_dilation_config=mask_dilation_config,
        )
        for s in val_scene_cfgs
    ]
    val_scene_names = [s.get('name', s['root']) for s in val_scene_cfgs]

    # 每个场景类别独立，label idx 在场景内自洽，无需全局映射
    for name, count, categories in zip(
        train_dataset.scene_names,
        train_dataset.scene_sample_counts,
        train_dataset.scene_categories,
    ):
        logger.info(f"  [train] {name}: {count} 样本, 类别: {categories}")
    for name, ds in zip(val_scene_names, val_datasets_raw):
        logger.info(f"  [val]   {name}: {len(ds)} 样本, 类别: {ds.categories}")

    # batch_size 为每张卡的 batch 大小，effective total = batch_size * world_size
    batch_size = supcon_config['supcon']['data']['batch_size']
    val_datasets = val_datasets_raw
    num_workers = supcon_config['supcon']['data']['num_workers']
    pin_memory = supcon_config['supcon']['data']['pin_memory']
    if rank == 0:
        logger.info(f"batch_size per GPU: {batch_size}, effective total batch_size: {batch_size * world_size} (world_size={world_size})")

    training_config = supcon_config['supcon']['training']
    total_steps = int(training_config.get('total_steps', 0))
    if total_steps <= 0:
        raise ValueError("training.total_steps 必须配置且 > 0；当前训练脚本不再支持 epochs")

    scene_sampling = training_config.get('scene_sampling', 'size_temperature')
    scene_sampling_alpha = float(training_config.get('scene_sampling_alpha', 0.5))
    train_sampler = SceneBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        image_sizes=image_sizes,
        total_steps=total_steps,
        scene_sampling=scene_sampling,
        scene_sampling_alpha=scene_sampling_alpha,
        seed=int(training_config.get('seed', 0)),
        rank=rank,
        world_size=world_size,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=multi_scale_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=supcon_config['supcon']['data'].get('prefetch_factor', 2) if num_workers > 0 else None,
    )
    logger.info(
        f"训练采样: total_steps={total_steps}, scene_sampling={scene_sampling}, "
        f"alpha={scene_sampling_alpha}, 单 DataLoader scenes={train_dataset.num_scenes}"
    )

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
    embedding_source = (
        supcon_config['supcon']
        .get('inference', {})
        .get('embedding_source', 'representations')
    )
    if embedding_source not in {'representations', 'projections'}:
        raise ValueError(f"inference.embedding_source 必须是 representations 或 projections，当前为 {embedding_source!r}")

    # 创建模型
    logger.info("创建模型...")
    model_config = supcon_config['supcon']['model']
    freeze_backbone = bool(training_config.get('freeze_backbone', False))

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
    if use_moco:
        _queue_size_cfg = moco_config.get('queue_size', 16384)
        min_queue_size, max_queue_size = parse_queue_size_config(_queue_size_cfg)
        emb_dim = model_config['embedding_dim']
        moco_queues = []
        for count in train_dataset.scene_sample_counts:
            adaptive_size = min(count * 2, max_queue_size)
            adaptive_size = max(adaptive_size, min_queue_size)
            adaptive_size = max((adaptive_size // batch_size) * batch_size, batch_size)
            q = MoCoQueue(queue_size=adaptive_size, embedding_dim=emb_dim)
            if queue_on_gpu:
                q = q.to(device)
            moco_queues.append(q)
        for name, q in zip(train_scene_names, moco_queues):
            logger.info(f"  MoCo queue [{name}]: size={q.queue_size}, device={'gpu' if queue_on_gpu else 'cpu'}")
    else:
        moco_queues = None

    # 创建Loss函数
    loss_config = supcon_config['supcon']['loss']
    criterion = build_loss_func(
        loss_config=loss_config,
        moco_config=moco_config,
        use_moco=use_moco,
        device=device,
        logger=logger,
    )

    # 创建优化器（backbone和projection head使用不同学习率）
    # 使用 raw_model 提取参数，确保与 DDP 内部参数一致
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
            T_max=total_steps,
            eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, total_steps // 3),
            gamma=0.1
        )

    # 恢复训练：只支持 total_steps checkpoint。
    start_step = 1
    best_margin = float('-inf')
    last_train_metrics = {'loss': 0.0, 'contrastive_loss': 0.0}
    if args.resume:
        logger.info(f"从checkpoint恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'global_step' not in checkpoint:
            raise ValueError("当前脚本只支持包含 global_step 的 step-only checkpoint")
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = int(checkpoint['global_step']) + 1
        best_margin = checkpoint.get('best_margin', float('-inf'))
        last_train_metrics['loss'] = float(checkpoint.get('train_loss', 0.0))

        if use_moco and moco_queues is not None and 'moco_queue_state' in checkpoint:
            for queue, state in zip(moco_queues, checkpoint['moco_queue_state']):
                queue.load_state_dict(state)
            logger.info("MoCo队列状态已恢复（per-scene）")

    if start_step > total_steps:
        logger.info(f"checkpoint 已到达 total_steps: start_step={start_step}, total_steps={total_steps}")

    train_sampler.set_start_step(start_step)

    def save_checkpoint(global_step: int, train_metrics: dict, val_metrics: Optional[dict] = None):
        if not is_main:
            return
        checkpoint_data = {
            'global_step': global_step,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_metrics.get('loss', 0.0),
            'best_margin': best_margin,
            'config': supcon_config,
        }
        if val_metrics is not None:
            checkpoint_data['val_loss'] = val_metrics['loss']
        if use_moco and moco_queues is not None:
            checkpoint_data['moco_queue_state'] = [q.state_dict() for q in moco_queues]
        torch.save(checkpoint_data, checkpoint_dir / "current_model.pth")
        torch.save(checkpoint_data, checkpoint_dir / f"checkpoint_step_{global_step}.pth")

    # 训练循环
    logger.info("开始 step-only 训练...")
    train_losses = []
    val_losses = []
    log_interval = int(training_config.get('log_interval', 50))
    eval_interval = int(training_config.get('eval_interval', 1000))
    save_interval_steps = int(training_config.get('save_interval_steps', 1000))
    recent_losses = deque(maxlen=max(1, log_interval))
    scene_step_counts = defaultdict(int)
    train_iter = iter(train_dataloader)
    pbar = tqdm(range(start_step, total_steps + 1), desc="Training", disable=not is_main)

    for global_step in pbar:
        batch = next(train_iter)
        scene_idx = int(batch['scene_idx'][0].item())
        scene_name = train_scene_names[scene_idx]
        scene_step_counts[scene_idx] += 1

        scene_queue = None
        if moco_queues is not None:
            scene_queue = moco_queues[scene_idx]

        train_metrics = train_one_batch(
            model=model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            global_step=global_step,
            use_moco=use_moco,
            moco_queue=scene_queue,
        )
        last_train_metrics = train_metrics

        scheduler.step()
        recent_losses.append(train_metrics['loss'])
        window_loss = sum(recent_losses) / len(recent_losses)
        stage_gate_weights = get_stage_gate_weights(raw_model, use_moco)

        if is_main:
            pbar.set_postfix({
                'loss': f"{train_metrics['loss']:.4f}",
                'scene': scene_idx,
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })

        should_log = global_step == start_step or global_step % log_interval == 0
        if should_log and is_main:
            train_losses.append(window_loss)
            log_msg = (
                f"Step {global_step}/{total_steps}: scene={scene_name}, "
                f"loss={train_metrics['loss']:.4f}, window_loss={window_loss:.4f}, "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )
            if use_moco:
                if 'pos_loss' in train_metrics:
                    log_msg += f", pos_loss={train_metrics['pos_loss']:.4f}"
                if 'neg_loss' in train_metrics:
                    log_msg += f", neg_loss={train_metrics['neg_loss']:.4f}"
                if moco_queues is not None:
                    q = moco_queues[scene_idx]
                    log_msg += f", moco_queue_full={q.is_full()}"
            if stage_gate_weights is not None:
                gate_msg = ", ".join(f"{k}={v:.3f}" for k, v in stage_gate_weights.items())
                log_msg += f", stage_gate=[{gate_msg}]"
            scene_stats = ", ".join(
                f"{train_scene_names[i]}={scene_step_counts[i]}"
                for i in range(len(train_scene_names))
                if scene_step_counts[i] > 0
            )
            log_msg += f"\n  scene_steps: {scene_stats}"
            logger.info(log_msg)

        val_metrics = None
        val_metrics_per_scene = []
        should_eval = args.use_eval and val_dataloaders and eval_interval > 0 and global_step % eval_interval == 0
        if should_eval:
            for val_scene_idx, (val_scene_name, val_dl) in enumerate(zip(val_scene_names, val_dataloaders)):
                if is_main:
                    logger.info(f"验证场景({val_scene_idx + 1}/{len(val_scene_names)}): {val_scene_name}")
                vm = validate(
                    raw_model,
                    val_dl,
                    criterion,
                    device,
                    use_moco=use_moco,
                    loss_temperature=loss_config['supcon']['temperature'],
                    embedding_source=embedding_source,
                    show_progress=is_main,
                )
                val_metrics_per_scene.append((val_scene_name, vm))
            if is_main:
                val_metrics = {
                    'loss': sum(m['loss'] for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'margin_pos_sim': sum(m.get('margin_pos_sim', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'margin_neg_sim': sum(m.get('margin_neg_sim', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'margin': sum(m.get('margin', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'knn_accuracy': sum(m.get('knn_accuracy', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'projection_margin': sum(m.get('projection_margin', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'projection_knn_accuracy': sum(m.get('projection_knn_accuracy', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                }
                best_margin = max(best_margin, val_metrics.get('margin', float('-inf')))
                val_losses.append(val_metrics['loss'])
                logger.info(
                    f"Step {global_step} val: loss={val_metrics['loss']:.4f}, "
                    f"Margin={val_metrics.get('margin', 0):.4f}, kNN={val_metrics.get('knn_accuracy', 0):.4f}, "
                    f"ProjMargin={val_metrics.get('projection_margin', 0):.4f}, "
                    f"ProjKNN={val_metrics.get('projection_knn_accuracy', 0):.4f}"
                )

        if is_main and supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
            log_dict = {
                'global_step': global_step,
                'train_loss': train_metrics['loss'],
                'train_window_loss': window_loss,
                'train_contrastive_loss': train_metrics.get('contrastive_loss', 0.0),
                'learning_rate': scheduler.get_last_lr()[0],
                'scene_idx': scene_idx,
                'scene_steps/current': scene_step_counts[scene_idx],
            }
            if use_moco:
                if 'pos_loss' in train_metrics:
                    log_dict['train_pos_loss'] = train_metrics['pos_loss']
                if 'neg_loss' in train_metrics:
                    log_dict['train_neg_loss'] = train_metrics['neg_loss']
                if moco_queues is not None:
                    log_dict[f'moco_queue_full/{scene_name}'] = float(moco_queues[scene_idx].is_full())
            for i, count in scene_step_counts.items():
                log_dict[f'scene_steps/{train_scene_names[i]}'] = count
            if stage_gate_weights is not None:
                for key, value in stage_gate_weights.items():
                    log_dict[f'stage_gate/{key}'] = value
            if val_metrics is not None:
                log_dict['val_loss'] = val_metrics['loss']
                log_dict['val_margin'] = val_metrics.get('margin', 0)
                log_dict['val_knn_accuracy'] = val_metrics.get('knn_accuracy', 0)
                log_dict['val_projection_margin'] = val_metrics.get('projection_margin', 0)
                log_dict['val_projection_knn_accuracy'] = val_metrics.get('projection_knn_accuracy', 0)
                for sn, vm in val_metrics_per_scene:
                    log_dict[f'val_loss/{sn}'] = vm['loss']
                    log_dict[f'val_margin/{sn}'] = vm.get('margin', 0)
                    log_dict[f'val_knn_accuracy/{sn}'] = vm.get('knn_accuracy', 0)
                    log_dict[f'val_projection_margin/{sn}'] = vm.get('projection_margin', 0)
                    log_dict[f'val_projection_knn_accuracy/{sn}'] = vm.get('projection_knn_accuracy', 0)
            wandb.log(log_dict, step=global_step)

        if save_interval_steps > 0 and global_step % save_interval_steps == 0:
            if is_ddp:
                dist.barrier()
            save_checkpoint(global_step, train_metrics, val_metrics)
            if is_ddp:
                dist.barrier()

    # 绘制损失曲线（仅主进程）
    if is_ddp:
        dist.barrier()
    if is_main:
        if total_steps >= start_step:
            save_checkpoint(total_steps, last_train_metrics)
        plot_loss_curve(
            train_losses,
            val_losses if args.use_eval else [],
            save_path=str(checkpoint_dir / "loss_curve.png"),
            title="SupCon Training Loss",
        )
        final_model_path = checkpoint_dir / "final_model.pth"
        torch.save({'model_state_dict': raw_model.state_dict()}, final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")
    if is_ddp:
        dist.barrier()

    logger.info("训练完成！")
    if is_main and supcon_config['supcon']['output'].get('use_wandb', False) and wandb is not None:
        wandb.finish()

    # 清理分布式进程组
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
