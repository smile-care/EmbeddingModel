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
        min_mask_weight: floor value for mask weights to avoid zero-weight
                         pooling on fully-background crops.
    """

    def __init__(
        self,
        cls_weight: float = 0.3,
        min_mask_weight: float = 0.02,
    ):
        super().__init__()
        self.cls_weight      = cls_weight
        self.min_mask_weight = min_mask_weight

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
        B, N, D = patch_tokens.shape
        nh = nw = int(N ** 0.5)

        # Resize mask to patch grid resolution
        patch_mask    = F.interpolate(mask, size=(nh, nw), mode="nearest")
        patch_mask    = torch.clamp(patch_mask, self.min_mask_weight, 1.0)  # (B, 1, nh, nw)
        patch_weights = patch_mask.view(B, N, 1)                            # (B, N, 1)

        # Mask-weighted average pooling
        weighted_sum = (patch_tokens * patch_weights).sum(dim=1)            # (B, D)
        weight_total = patch_weights.sum(dim=1) + 1e-8                      # (B, 1)
        masked_pool  = weighted_sum / weight_total                           # (B, D)

        # Blend CLS (global) and masked patch pooling (foreground-aware)
        return self.cls_weight * cls_token + (1.0 - self.cls_weight) * masked_pool  # (B, D)
