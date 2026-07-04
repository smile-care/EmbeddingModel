"""Unified SupCon/MoCo model construction.

This is the single place where the embedding model (backbone + mask-weighted
feature fusion + projection head, optionally wrapped in MoCo) is built from a config dict. Both
the standalone trainer (``scripts/train_supcon.py``) and the platform
training/inference entry points (``data_cluster.dl.trainer`` /
``data_cluster.dl.inference``) call ``build_supcon_model`` so the architecture
stays in sync.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from embedding_model.supcon.models.backbone.dinov3_convnext import DINOv3ConvNextConfig
from embedding_model.supcon.models.backbone.dinov3_vit import DINOv3ViTConfig
from embedding_model.supcon.models.convnext_model import ConvNeXtModel
from embedding_model.supcon.models.moco_model import MoCoModel
from embedding_model.supcon.models.moco_queue import MoCoQueue
from embedding_model.supcon.models.vit_model import ViTModel
from embedding_model.utils.config_loader import load_config


def _convnext_build_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve ConvNeXt-specific keys from flat or nested supcon config."""
    cnx = model_config.get("convnext")
    cnx = cnx if isinstance(cnx, dict) else {}

    use_layers = model_config.get("use_layers")
    if use_layers is None:
        use_layers = cnx.get("use_layers", [1, 2, 3])

    embedding_dim = int(model_config.get("embedding_dim", 128))
    fusion_dim = cnx.get("fusion_dim", model_config.get("fusion_dim", embedding_dim))
    fusion_dim = int(fusion_dim) if fusion_dim is not None else embedding_dim
    projection_hidden_dim = cnx.get("projection_hidden_dim", model_config.get("projection_hidden_dim"))
    projection_hidden_dim = int(projection_hidden_dim) if projection_hidden_dim is not None else None

    return {
        "use_layers": list(use_layers),
        "fusion_dim": fusion_dim,
        "projection_hidden_dim": projection_hidden_dim,
        "mask_gating": dict(cnx.get("mask_gating", {})),
        "pooling": dict(cnx.get("pooling", {"mode": "fg_only"})),
    }


def _vit_build_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve ViT-specific keys from nested supcon config."""
    vit = model_config.get("vit")
    vit = vit if isinstance(vit, dict) else {}
    projection_hidden_dim = vit.get("projection_hidden_dim", model_config.get("projection_hidden_dim"))
    projection_hidden_dim = int(projection_hidden_dim) if projection_hidden_dim is not None else None
    return {
        "cls_weight": float(vit.get("cls_weight", 0.3)),
        "projection_hidden_dim": projection_hidden_dim,
    }


def _resolve_queue_size(moco_config: dict[str, Any]) -> int:
    raw = moco_config.get("queue_size", 16384)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, (list, tuple)) and raw:
        try:
            return int(max(int(x) for x in raw))
        except (TypeError, ValueError):
            pass
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 16384


def extract_model_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Parse ``model_state_dict`` / ``state_dict`` / ``model`` from a checkpoint."""
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint 必须是 dict，当前类型: {type(checkpoint)}")

    for key in ("model_state_dict", "state_dict", "model"):
        state = checkpoint.get(key)
        if isinstance(state, dict) and state:
            return state

    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        sample_keys = list(checkpoint.keys())[:5]
        if any(
            k.startswith(prefix)
            for k in sample_keys
            for prefix in (
                "backbone.",
                "query_encoder.",
                "momentum_encoder.",
                "feature_fusion.",
                "projection_head.",
            )
        ):
            return checkpoint  # type: ignore[return-value]

    raise ValueError(
        "无法在 checkpoint 中找到 model_state_dict / state_dict / model"
    )


def is_full_supcon_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """True when the checkpoint includes trained fusion/head (not backbone-only)."""
    return any(
        "feature_fusion." in k or "projection_head." in k
        for k in state_dict
    )


def normalize_state_dict_for_supcon_model(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map MoCo ``query_encoder.*`` keys to ConvNeXtModel/ViTModel layout; drop momentum copy."""
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("momentum_encoder."):
            continue
        if key.startswith("query_encoder."):
            normalized[key[len("query_encoder."):]] = value
        else:
            normalized[key] = value
    return normalized


def is_full_supcon_checkpoint(path: str | Path) -> bool:
    """Peek at a .pth file to see if it contains fusion/head weights."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = extract_model_state_dict(ckpt)
    return is_full_supcon_state_dict(state)


def resolve_backbone_config(
    model_config: dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> tuple[DINOv3ConvNextConfig | DINOv3ViTConfig, str | None]:
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
    model_type = backbone_cfg_dict.get("model_type", "dinov3_convnext")
    if model_type == "dinov3_vit":
        backbone_cfg = DINOv3ViTConfig.from_dict(backbone_cfg_dict)
    elif model_type == "dinov3_convnext":
        backbone_cfg = DINOv3ConvNextConfig.from_dict(backbone_cfg_dict)
    else:
        raise ValueError(f"不支持的 backbone model_type: {model_type!r}")

    ckpt_path = model_config.get("pretrained_path")
    if ckpt_path is not None:
        ckpt_path = str(ckpt_path)

    if logger is not None:
        logger.info(f"Backbone 配置: {backbone_config_path}")
        logger.info(f"Backbone 类型: {model_type}")
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
    """Build a ConvNeXtModel/ViTModel (or MoCoModel + queue) from config.

    Returns ``(model, moco_queue)`` where ``moco_queue`` is ``None`` unless
    ``use_moco`` is True.  The backbone's own pretrained weights (if any) are
    loaded at construction time from ``model_config['pretrained_path']``.
    """
    moco_config = moco_config or {}
    backbone_cfg, ckpt_path = resolve_backbone_config(model_config, logger=logger)
    is_vit = isinstance(backbone_cfg, DINOv3ViTConfig)

    common_kwargs = dict(
        backbone_cfg=backbone_cfg,
        ckpt_path=ckpt_path,
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        image_size=image_size,
        freeze_backbone=freeze_backbone,
    )
    if is_vit:
        common_kwargs.update(_vit_build_kwargs(model_config))
    else:
        common_kwargs.update(_convnext_build_kwargs(model_config))

    if use_moco:
        if logger is not None:
            logger.info("使用 MoCo 模型（动量对比学习）")
        momentum = float(moco_config.get("momentum", 0.999))
        model = MoCoModel(momentum=momentum, **common_kwargs)
        queue_size = _resolve_queue_size(moco_config)
        moco_queue = MoCoQueue(
            queue_size=queue_size,
            embedding_dim=int(model_config.get("embedding_dim", 128)),
        )
        if device is not None:
            model = model.to(device)
            moco_queue = moco_queue.to(device)
        return model, moco_queue

    if logger is not None:
        logger.info(f"使用标准 SupCon 模型（{'ViTModel' if is_vit else 'ConvNeXtModel'}）")
    model_cls = ViTModel if is_vit else ConvNeXtModel
    model = model_cls(**common_kwargs)
    if device is not None:
        model = model.to(device)
    return model, None
