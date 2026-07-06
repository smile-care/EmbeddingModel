"""Reusable helpers for SupCon training scripts."""
import os
import re
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.embedding_model.supcon.models.backbone.dinov3_convnext import DINOv3ConvNextConfig
from src.embedding_model.supcon.models.backbone.dinov3_vit import DINOv3ViTConfig
from src.embedding_model.supcon.models.convnext_model import ConvNeXtModel
from src.embedding_model.supcon.models.losses import SupervisedContrastiveLoss
from src.embedding_model.supcon.models.moco_loss import MoCoLoss
from src.embedding_model.supcon.models.moco_model import MoCoModel
from src.embedding_model.supcon.models.moco_queue import MoCoQueue
from src.embedding_model.supcon.models.vit_model import ViTModel
from src.embedding_model.utils.metrics import knn_evaluation, similarity_distribution_stats


def sanitize_wandb_project_name(project_name: str, default: str = "industrial-supcon") -> str:
    """Clean W&B project names that contain unsupported characters."""
    if not project_name:
        return default

    sanitized = re.sub(r"[\/\\#\?%:]+", "-", str(project_name)).strip()
    sanitized = re.sub(r"\s+", "-", sanitized)
    sanitized = sanitized.strip("-_.")
    return sanitized or default


def parse_queue_size_config(queue_size_config) -> tuple[int, int]:
    """Parse fixed or [min, max] MoCo queue size config."""
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
    """Initialize DDP and return (local_rank, rank, world_size)."""
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
    """Train one SupCon/MoCo step."""
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
    """Validate and compute embedding similarity/kNN metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    nan_embedding_count = 0
    nan_loss_count = 0

    all_projections = []
    all_app_embeddings = []
    all_labels = []
    temp_criterion = None
    if use_moco:
        temp_criterion = SupervisedContrastiveLoss(temperature=loss_temperature).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", disable=not show_progress):
            view1_images = batch['view1_image'].to(device)
            view1_masks = batch['view1_mask'].to(device)
            labels = batch['label'].to(device)

            if use_moco:
                outputs1 = model(view1_images, view1_masks, mode='query', return_features=False)
            else:
                outputs1 = model(view1_images, view1_masks, return_features=False)
            projections1 = outputs1['projections']
            app_embeddings1 = outputs1[embedding_source]

            projections1 = F.normalize(projections1, dim=1, p=2, eps=1e-8)
            app_embeddings1 = F.normalize(app_embeddings1, dim=1, p=2, eps=1e-8)

            if (torch.isnan(projections1).any() or torch.isinf(projections1).any() or
                torch.isnan(app_embeddings1).any() or torch.isinf(app_embeddings1).any()):
                nan_embedding_count += 1
                skipped_batches += 1
                if nan_embedding_count <= 5:
                    print("警告：验证时模型输出包含NaN或Inf，跳过此batch")
                continue

            if use_moco:
                loss = temp_criterion(projections1, labels)
            else:
                loss = criterion(projections1, labels)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
                nan_loss_count += 1
                skipped_batches += 1
                if nan_loss_count <= 5:
                    print("警告：验证时loss为NaN或Inf，跳过此batch")
                continue

            all_projections.append(projections1.cpu())
            all_app_embeddings.append(app_embeddings1.cpu())
            all_labels.append(labels.cpu())

            total_loss += loss.item()
            num_batches += 1

    if skipped_batches > 0:
        print(f"验证统计: 跳过{skipped_batches}个batch (NaN embedding: {nan_embedding_count}, NaN loss: {nan_loss_count})")

    if len(all_projections) > 0:
        all_projections_tensor = torch.cat(all_projections, dim=0).to(device)
        all_app_embeddings_tensor = torch.cat(all_app_embeddings, dim=0).to(device)
        all_labels_tensor = torch.cat(all_labels, dim=0).to(device)

        sim_stats = similarity_distribution_stats(all_app_embeddings_tensor, all_labels_tensor)
        margin_pos_sim = sim_stats['pos_sim']
        margin_neg_sim = sim_stats['neg_sim']
        margin = sim_stats['margin']
        knn_stats = knn_evaluation(all_app_embeddings_tensor, all_labels_tensor, k=10)
        knn_accuracy = knn_stats.get('knn_accuracy', 0.0)

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
        'margin_pos_sim': margin_pos_sim,
        'margin_neg_sim': margin_neg_sim,
        'margin': margin,
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
    log_fn: Optional[Callable[[str], None]] = None,
) -> nn.Module:
    """Build SupCon or MoCo model from config."""
    backbone_name = model_config.get('backbone')
    if not backbone_name:
        raise ValueError("model_config 中必须指定 backbone 字段")
    backbone_cfg_path = f"configs/backbone/{backbone_name}.yaml"
    backbone_ckpt_path = f"pretrain_ckpts/{backbone_name}.pth"
    if log_fn is not None:
        log_fn(f"Backbone: {backbone_name}")

    with open(backbone_cfg_path) as f:
        raw_cfg = yaml.safe_load(f)
    model_type = raw_cfg.get('model_type', 'dinov3_convnext')
    is_vit = model_type == 'dinov3_vit'

    if is_vit:
        backbone_cfg = DINOv3ViTConfig.from_dict(raw_cfg)
        if log_fn is not None:
            log_fn("管线: ViT → MaskWeightedPooling → ProjectionHead")
    else:
        backbone_cfg = DINOv3ConvNextConfig.from_dict(raw_cfg)
        if log_fn is not None:
            log_fn("管线: ConvNeXt → MultiStageMaskPooling → ProjectionHead")

    common_kwargs = dict(
        backbone_cfg=backbone_cfg,
        ckpt_path=backbone_ckpt_path,
        embedding_dim=model_config['embedding_dim'],
        image_size=image_size,
        freeze_backbone=freeze_backbone,
    )

    if is_vit:
        vit_cfg = model_config.get('vit', {})
        common_kwargs['cls_weight'] = vit_cfg.get('cls_weight', 0.3)
        common_kwargs['projection_hidden_dim'] = vit_cfg.get(
            'projection_hidden_dim',
            model_config.get('projection_hidden_dim'),
        )
    else:
        cnx_cfg = model_config.get('convnext', {})
        common_kwargs.update(dict(
            use_layers=cnx_cfg.get('use_layers', [1, 2, 3]),
            fusion_dim=cnx_cfg.get('fusion_dim', model_config.get('embedding_dim')),
            projection_hidden_dim=cnx_cfg.get(
                'projection_hidden_dim',
                model_config.get('projection_hidden_dim'),
            ),
            mask_gating=cnx_cfg.get('mask_gating', {}),
            pooling=cnx_cfg.get('pooling', {'mode': 'fg_only'}),
        ))

    if use_moco:
        momentum = moco_config.get('momentum', 0.999)
        if log_fn is not None:
            log_fn(f"MoCo 动量对比学习，momentum={momentum}")
        return MoCoModel(**common_kwargs, momentum=momentum).to(device)

    model_cls = ViTModel if is_vit else ConvNeXtModel
    if log_fn is not None:
        log_fn(f"标准 SupCon 模型: {model_cls.__name__}")
    return model_cls(**common_kwargs).to(device)


def build_loss_func(
    loss_config: dict,
    moco_config: dict,
    use_moco: bool,
    device: torch.device,
    log_fn: Optional[Callable[[str], None]] = None,
) -> nn.Module:
    """Build SupCon or MoCo criterion from config."""
    if use_moco:
        moco_loss_type = moco_config.get('loss_type', 'supervised')
        neg_weight = loss_config.get('neg_weight', 0.5)
        margin = loss_config.get('margin', 0.0)
        if log_fn is not None:
            log_fn(f"使用MoCoLoss（类型: {moco_loss_type}）")
            log_fn(f"  负样本惩罚权重: {neg_weight}, margin: {margin}")
        return MoCoLoss(
            temperature=loss_config['supcon']['temperature'],
            loss_type=moco_loss_type,
            neg_weight=neg_weight,
            margin=margin,
        ).to(device)

    if log_fn is not None:
        log_fn("使用SupervisedContrastiveLoss（标准SupCon模式，仅同label为positive）")
    return SupervisedContrastiveLoss(
        temperature=loss_config['supcon']['temperature'],
    ).to(device)


def get_stage_gate_weights(raw_model: nn.Module, use_moco: bool) -> Optional[dict]:
    """Return ConvNeXt FeatureFusion stage gate weights if enabled."""
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
