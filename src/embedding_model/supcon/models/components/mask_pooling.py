"""Mask-weighted pooling for ViT patch token sequences."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskWeightedPooling(nn.Module):
    """Fuse CLS token and mask-weighted patch pooling into a single vector.

    Outputs a raw D-dimensional vector (same as ViT hidden_size) with no
    projection — dimensionality reduction is fully delegated to ProjectionHead.

    Args:
        cls_weight:      α in  output = α*CLS + (1-α)*masked_patch_pooling.
                         0.0 → pure patch pooling, 1.0 → pure CLS.
    """

    def __init__(
        self,
        cls_weight: float = 0.3,
    ):
        super().__init__()
        if not 0.0 <= cls_weight <= 1.0:
            raise ValueError(f"cls_weight 必须在 [0, 1]，当前为 {cls_weight}")
        self.cls_weight = cls_weight

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_token:    (B, D)
            patch_tokens: (B, N, D)  N = nh * nw, no CLS/register tokens
            mask:         (B, 1, H, W)  foreground mask, values in [0, 1]

        Returns:
            fused: (B, D)  — same dimension as ViT hidden_size
        """
        self._validate_inputs(cls_token, patch_tokens, mask)
        B, N, D = patch_tokens.shape
        nh = nw = int(N ** 0.5)

        patch_mask = F.interpolate(mask.float(), size=(nh, nw), mode="area")
        patch_weights = torch.clamp(patch_mask, 0.0, 1.0).view(B, N, 1)

        weighted_sum = (patch_tokens * patch_weights).sum(dim=1)
        weight_total = patch_weights.sum(dim=1)
        masked_pool = weighted_sum / weight_total.clamp_min(1e-8)
        global_pool = patch_tokens.mean(dim=1)
        has_foreground = (weight_total > 1e-6).expand_as(masked_pool)
        pooled = torch.where(has_foreground, masked_pool, global_pool)

        return self.cls_weight * cls_token + (1.0 - self.cls_weight) * pooled

    def _validate_inputs(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if cls_token.ndim != 2:
            raise ValueError(f"cls_token 必须是 (B, D)，当前 shape={tuple(cls_token.shape)}")
        if patch_tokens.ndim != 3:
            raise ValueError(f"patch_tokens 必须是 (B, N, D)，当前 shape={tuple(patch_tokens.shape)}")
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"mask 必须是 (B, 1, H, W)，当前 shape={tuple(mask.shape)}")
        if cls_token.shape[0] != patch_tokens.shape[0] or cls_token.shape[0] != mask.shape[0]:
            raise ValueError("cls_token、patch_tokens、mask 的 batch size 必须一致")
        if cls_token.shape[1] != patch_tokens.shape[2]:
            raise ValueError(
                f"cls_token dim ({cls_token.shape[1]}) 与 patch_tokens dim ({patch_tokens.shape[2]}) 不一致"
            )

        num_patches = patch_tokens.shape[1]
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"patch_tokens 的 N 必须是平方数，当前 N={num_patches}")
