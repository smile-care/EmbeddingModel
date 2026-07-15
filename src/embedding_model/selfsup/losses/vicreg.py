"""Distributed VICReg objective."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    from torch.distributed.nn.functional import all_gather

    return torch.cat(list(all_gather(tensor)), dim=0)


class VICRegLoss(nn.Module):
    """Invariance, variance and covariance regularization without negative pairs."""

    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        variance_target: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.invariance_weight = float(invariance_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.variance_target = float(variance_target)
        self.eps = float(eps)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> dict[str, torch.Tensor]:
        if view1.shape != view2.shape or view1.ndim != 2:
            raise ValueError(
                f"VICReg 输入必须是相同 shape 的二维张量，当前为 {view1.shape} 和 {view2.shape}"
            )

        # Covariance/variance statistics are too sensitive for fp16 accumulation.
        view1 = view1.float()
        view2 = view2.float()
        invariance = F.mse_loss(view1, view2)
        global_view1 = _gather_with_grad(view1)
        global_view2 = _gather_with_grad(view2)
        variance = self._variance_loss(global_view1) + self._variance_loss(global_view2)
        covariance = self._covariance_loss(global_view1) + self._covariance_loss(global_view2)
        total = (
            self.invariance_weight * invariance
            + self.variance_weight * variance
            + self.covariance_weight * covariance
        )
        average_std = 0.5 * (
            torch.sqrt(global_view1.var(dim=0, unbiased=False) + self.eps).mean()
            + torch.sqrt(global_view2.var(dim=0, unbiased=False) + self.eps).mean()
        )
        return {
            "loss": total,
            "invariance_loss": invariance,
            "variance_loss": variance,
            "covariance_loss": covariance,
            "average_std": average_std,
        }

    def _variance_loss(self, values: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(values.var(dim=0, unbiased=False) + self.eps)
        return F.relu(self.variance_target - std).mean()

    @staticmethod
    def _covariance_loss(values: torch.Tensor) -> torch.Tensor:
        sample_count, feature_dim = values.shape
        if sample_count < 2:
            return values.sum() * 0.0
        centered = values - values.mean(dim=0)
        covariance = centered.T @ centered / (sample_count - 1)
        diagonal = torch.diagonal(covariance)
        off_diagonal_squared_sum = covariance.square().sum() - diagonal.square().sum()
        return off_diagonal_squared_sum / feature_dim
