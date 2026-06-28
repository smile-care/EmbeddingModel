"""Unified SupCon/MoCo model construction.

This is the single place where the embedding model (backbone + FPN + fusion +
projection head, optionally wrapped in MoCo) is built from a config dict.  Both
training (``SupconTrainer``) and inference (``compute_supcon_embeddings``) call
``build_supcon_model`` so the architecture stays in sync.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from embedding_model.supcon.models.backbone.dinov3_convnext import DINOv3ConvNextConfig
from embedding_model.supcon.models.convnext_model import ConvNeXtModel
from embedding_model.supcon.models.moco_model import MoCoModel
from embedding_model.supcon.models.moco_queue import MoCoQueue
from embedding_model.utils.config_loader import load_config


def _projection_hidden_dims(model_config: dict[str, Any]) -> list[int]:
    """Support both the nested (``projection_head.hidden_dims``) and the flat
    (``projection_hidden_dims``) config shapes."""
    head = model_config.get("projection_head")
    if isinstance(head, dict) and head.get("hidden_dims") is not None:
        return list(head["hidden_dims"])
    return list(model_config.get("projection_hidden_dims", [256, 128]))


def resolve_backbone_config(
    model_config: dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> tuple[DINOv3ConvNextConfig, str | None]:
    """Resolve the backbone architecture config and pretrained checkpoint path.

    Supported keys in ``model_config``:
      * ``backbone``            - backbone name -> ``configs/backbone/{name}.yaml``
      * ``backbone_config_path``- explicit backbone YAML path (takes precedence)
      * ``pretrained_path``     - backbone pretrained weights (.pth)
    """
    backbone_config_path = model_config.get("backbone_config_path")
    if not backbone_config_path:
        backbone_name = model_config.get("backbone", "convnext_small")
        backbone_config_path = f"configs/backbone/{backbone_name}.yaml"

    backbone_cfg_dict = load_config(backbone_config_path)
    backbone_cfg = DINOv3ConvNextConfig.from_dict(backbone_cfg_dict)

    ckpt_path = model_config.get("pretrained_path")
    if ckpt_path is not None:
        ckpt_path = str(ckpt_path)

    if logger is not None:
        logger.info(f"Backbone 配置: {backbone_config_path}")
        if ckpt_path:
            logger.info(f"预训练权重(backbone): {ckpt_path}")

    return backbone_cfg, ckpt_path


def build_supcon_model(
    model_config: dict[str, Any],
    *,
    image_size: int,
    use_moco: bool = False,
    moco_config: Optional[dict[str, Any]] = None,
    freeze_backbone: bool = False,
    device: torch.device | str | None = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[torch.nn.Module, Optional[MoCoQueue]]:
    """Build a ConvNeXtModel (or MoCoModel + queue) from a model config.

    Returns ``(model, moco_queue)`` where ``moco_queue`` is ``None`` unless
    ``use_moco`` is True.  The backbone's own pretrained weights (if any) are
    loaded at construction time from ``model_config['pretrained_path']``.
    """
    moco_config = moco_config or {}
    backbone_cfg, ckpt_path = resolve_backbone_config(model_config, logger=logger)

    seg_config = model_config.get("segmentation", {}) or {}
    enable_segmentation = bool(seg_config.get("enabled", True))
    seg_layer_idx = int(seg_config.get("layer_idx", 0))

    common_kwargs = dict(
        backbone_cfg=backbone_cfg,
        ckpt_path=ckpt_path,
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        projection_hidden_dims=_projection_hidden_dims(model_config),
        image_size=image_size,
        freeze_backbone=freeze_backbone,
        use_layers=model_config.get("use_layers", None),
        fpn_out_channels=int(model_config.get("fpn_out_channels", 256)),
        fusion_dim=int(model_config.get("fusion_dim", 512)),
        enable_segmentation=enable_segmentation,
        seg_layer_idx=seg_layer_idx,
    )

    if use_moco:
        if logger is not None:
            logger.info("使用 MoCo 模型（动量对比学习）")
        momentum = float(moco_config.get("momentum", 0.999))
        model = MoCoModel(momentum=momentum, **common_kwargs)
        queue_size = int(moco_config.get("queue_size", 16384))
        moco_queue = MoCoQueue(
            queue_size=queue_size,
            embedding_dim=int(model_config.get("embedding_dim", 128)),
        )
        if device is not None:
            model = model.to(device)
            moco_queue = moco_queue.to(device)
        return model, moco_queue

    if logger is not None:
        logger.info("使用标准 SupCon 模型（ConvNeXtModel）")
    model = ConvNeXtModel(**common_kwargs)
    if device is not None:
        model = model.to(device)
    return model, None
