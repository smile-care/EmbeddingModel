"""Mask-aware encoders built on the project's shared DINOv3 backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from src.embedding_model.supcon.models.backbone.dinov3_convnext import (
    DINOv3ConvNext,
    DINOv3ConvNextConfig,
)
from src.embedding_model.supcon.models.backbone.dinov3_vit import DINOv3ViT, DINOv3ViTConfig

from .pooling import ConvNeXtMaskAggregator, ViTMaskAggregator


class MaskAwareEncoder(nn.Module):
    """Combine a shared DINOv3 backbone with a selfsup-owned mask aggregator."""

    def __init__(self, backbone: nn.Module, aggregator: nn.Module, backbone_type: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.backbone_type = backbone_type

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "convnext":
            features = self.backbone(images, output_hidden_states=True)
            return self.aggregator(features, masks)
        cls_token, patch_tokens = self.backbone(images, output_hidden_states=True)
        return self.aggregator(cls_token, patch_tokens, masks)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable


def build_mask_aware_encoder(
    model_config: dict[str, Any],
    *,
    project_root: str | Path,
    load_pretrained: bool = True,
) -> MaskAwareEncoder:
    """Build either ConvNeXt Base or ViT Base from the common backbone implementation."""
    project_root = Path(project_root)
    backbone_name = str(model_config.get("backbone", ""))
    if not backbone_name:
        raise ValueError("selfsup.model.backbone 不能为空")

    backbone_config_path = project_root / "configs" / "backbone" / f"{backbone_name}.yaml"
    if not backbone_config_path.is_file():
        raise FileNotFoundError(f"Backbone 配置不存在: {backbone_config_path}")
    with backbone_config_path.open("r", encoding="utf-8") as file:
        raw_backbone_config = yaml.safe_load(file)

    pretrained_path = model_config.get("pretrained_path")
    if pretrained_path is None:
        pretrained_path = project_root / "pretrain_ckpts" / f"{backbone_name}.pth"
    else:
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.is_absolute():
            pretrained_path = project_root / pretrained_path
    checkpoint_path = str(pretrained_path) if load_pretrained else None
    if load_pretrained and not pretrained_path.is_file():
        raise FileNotFoundError(f"Backbone 预训练权重不存在: {pretrained_path}")

    representation_dim = int(model_config["representation_dim"])
    model_type = raw_backbone_config.get("model_type")
    if model_type == "dinov3_convnext":
        backbone_config = DINOv3ConvNextConfig.from_dict(raw_backbone_config)
        backbone = DINOv3ConvNext(cfg=backbone_config, ckpt_path=checkpoint_path)
        convnext_config = model_config.get("convnext", {})
        pooling_config = convnext_config.get("pooling", {})
        aggregator = ConvNeXtMaskAggregator(
            feature_dims=backbone_config.hidden_sizes,
            representation_dim=representation_dim,
            use_layers=convnext_config.get("use_layers", [1, 2, 3]),
            pooling_mode=pooling_config.get("mode", "fg_context_delta"),
            context_dilation=int(pooling_config.get("context_dilation", 2)),
            stage_gate=bool(pooling_config.get("stage_gate", True)),
        )
        return MaskAwareEncoder(backbone, aggregator, backbone_type="convnext")

    if model_type == "dinov3_vit":
        backbone_config = DINOv3ViTConfig.from_dict(raw_backbone_config)
        backbone = DINOv3ViT(cfg=backbone_config, ckpt_path=checkpoint_path)
        vit_config = model_config.get("vit", {})
        aggregator = ViTMaskAggregator(
            hidden_dim=backbone_config.hidden_size,
            representation_dim=representation_dim,
            cls_weight=float(vit_config.get("cls_weight", 0.0)),
        )
        return MaskAwareEncoder(backbone, aggregator, backbone_type="vit")

    raise ValueError(f"不支持的 backbone model_type: {model_type!r}")
