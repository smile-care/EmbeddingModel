"""DINO-style centered teacher-student cross-entropy."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Match crossed student/teacher views and maintain a distributed teacher center."""

    def __init__(self, num_prototypes: int, center_momentum: float = 0.9) -> None:
        super().__init__()
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, num_prototypes))

    def forward(
        self,
        student_view1: torch.Tensor,
        student_view2: torch.Tensor,
        teacher_view1: torch.Tensor,
        teacher_view2: torch.Tensor,
        *,
        student_temperature: float,
        teacher_temperature: float,
    ) -> dict[str, torch.Tensor]:
        student1 = F.log_softmax(student_view1 / student_temperature, dim=-1)
        student2 = F.log_softmax(student_view2 / student_temperature, dim=-1)
        teacher1 = F.softmax((teacher_view1.detach() - self.center) / teacher_temperature, dim=-1)
        teacher2 = F.softmax((teacher_view2.detach() - self.center) / teacher_temperature, dim=-1)

        cross12 = -(teacher1 * student2).sum(dim=-1).mean()
        cross21 = -(teacher2 * student1).sum(dim=-1).mean()
        loss = 0.5 * (cross12 + cross21)
        self._update_center(torch.cat([teacher_view1.detach(), teacher_view2.detach()], dim=0))

        teacher_probabilities = torch.cat([teacher1, teacher2], dim=0)
        teacher_entropy = -(teacher_probabilities * teacher_probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
        prototype_usage = teacher_probabilities.mean(dim=0)
        usage_entropy = -(prototype_usage * prototype_usage.clamp_min(1e-12).log()).sum()
        return {
            "loss": loss,
            "teacher_entropy": teacher_entropy,
            "prototype_usage_entropy": usage_entropy,
            "max_prototype_probability": prototype_usage.max(),
        }

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_sum = teacher_logits.sum(dim=0, keepdim=True)
        batch_count = torch.tensor(
            [teacher_logits.shape[0]],
            device=teacher_logits.device,
            dtype=teacher_logits.dtype,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sum)
            dist.all_reduce(batch_count)
        batch_center = batch_sum / batch_count.clamp_min(1.0)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)
