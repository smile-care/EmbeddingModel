"""Self-supervised mask pooling modules, independent from the SupCon stack."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_average(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = F.interpolate(mask.float(), size=feature.shape[-2:], mode="area").clamp_(0.0, 1.0)
    weight_sum = weights.sum(dim=(2, 3))
    foreground = (feature * weights).sum(dim=(2, 3)) / weight_sum.clamp_min(1e-6)
    global_average = feature.mean(dim=(2, 3))
    return torch.where((weight_sum > 1e-6).expand_as(foreground), foreground, global_average)


class ConvNeXtMaskAggregator(nn.Module):
    """Fuse mask-aware foreground and local-context features across ConvNeXt stages."""

    def __init__(
        self,
        feature_dims: Sequence[int],
        representation_dim: int,
        use_layers: Sequence[int],
        pooling_mode: str = "fg_context_delta",
        context_dilation: int = 2,
        stage_gate: bool = True,
    ) -> None:
        super().__init__()
        if pooling_mode not in {"fg_only", "fg_context_delta"}:
            raise ValueError(f"不支持的 ConvNeXt pooling_mode: {pooling_mode!r}")
        self.use_layers = list(use_layers)
        if not self.use_layers:
            raise ValueError("ConvNeXt use_layers 不能为空")
        if any(index < 0 or index >= len(feature_dims) for index in self.use_layers):
            raise ValueError(f"use_layers 超出 backbone stage 范围: {self.use_layers}")
        if context_dilation < 1:
            raise ValueError("context_dilation 必须 >= 1")

        self.pooling_mode = pooling_mode
        self.context_dilation = int(context_dilation)
        pooled_multiplier = 1 if pooling_mode == "fg_only" else 3
        self.stage_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dims[index] * pooled_multiplier, representation_dim),
                    nn.LayerNorm(representation_dim),
                    nn.GELU(),
                )
                for index in self.use_layers
            ]
        )
        self.stage_logits = nn.Parameter(torch.zeros(len(self.use_layers))) if stage_gate else None
        self.fusion = nn.Sequential(
            nn.Linear(representation_dim * len(self.use_layers), representation_dim),
            nn.LayerNorm(representation_dim),
            nn.GELU(),
        )

    def forward(self, features: Sequence[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        projected: list[torch.Tensor] = []
        for projection, layer_index in zip(self.stage_projections, self.use_layers):
            feature = features[layer_index]
            pooled = self._pool_feature(feature, mask)
            projected.append(projection(pooled))

        if self.stage_logits is not None:
            weights = torch.softmax(self.stage_logits, dim=0)
            projected = [weight * value for weight, value in zip(weights, projected)]
        return self.fusion(torch.cat(projected, dim=1))

    def _pool_feature(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        foreground = _masked_average(feature, mask)
        if self.pooling_mode == "fg_only":
            return foreground

        resized_mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="area").clamp_(0.0, 1.0)
        kernel_size = 2 * self.context_dilation + 1
        dilated = F.max_pool2d(
            resized_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=self.context_dilation,
        )
        context_mask = (dilated - resized_mask).clamp_(0.0, 1.0)
        context = _masked_average(feature, context_mask)
        return torch.cat([foreground, context, foreground - context], dim=1)


class ViTMaskAggregator(nn.Module):
    """Pool ViT patch tokens with a resized foreground mask and optional CLS mixing."""

    def __init__(self, hidden_dim: int, representation_dim: int, cls_weight: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= cls_weight <= 1.0:
            raise ValueError("cls_weight 必须在 [0, 1]")
        self.cls_weight = float(cls_weight)
        self.output = (
            nn.Identity()
            if hidden_dim == representation_dim
            else nn.Sequential(
                nn.Linear(hidden_dim, representation_dim),
                nn.LayerNorm(representation_dim),
                nn.GELU(),
            )
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_patches, _ = patch_tokens.shape
        grid_size = int(num_patches**0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"ViT patch 数量必须是平方数，当前为 {num_patches}")

        patch_mask = F.interpolate(mask.float(), size=(grid_size, grid_size), mode="area")
        patch_weights = patch_mask.clamp_(0.0, 1.0).reshape(batch_size, num_patches, 1)
        weight_sum = patch_weights.sum(dim=1)
        foreground = (patch_tokens * patch_weights).sum(dim=1) / weight_sum.clamp_min(1e-6)
        global_average = patch_tokens.mean(dim=1)
        foreground = torch.where((weight_sum > 1e-6).expand_as(foreground), foreground, global_average)
        pooled = self.cls_weight * cls_token + (1.0 - self.cls_weight) * foreground
        return self.output(pooled)
