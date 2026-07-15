"""EMA teacher management for queue-free DINO-style self-distillation."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMATeacher(nn.Module):
    """A non-trainable exponential moving average copy of the student network."""

    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.model = copy.deepcopy(student)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, student: nn.Module, momentum: float) -> None:
        for teacher_parameter, student_parameter in zip(
            self.model.parameters(),
            student.parameters(),
        ):
            teacher_parameter.mul_(momentum).add_(student_parameter.detach(), alpha=1.0 - momentum)

        for teacher_buffer, student_buffer in zip(
            self.model.buffers(),
            student.buffers(),
        ):
            teacher_buffer.copy_(student_buffer)

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(images, masks)
