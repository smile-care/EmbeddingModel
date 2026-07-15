"""Projection and prototype heads used only by self-supervised objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MaskAwareEncoder


class VICRegProjector(nn.Module):
    """Unnormalized MLP projector; VICReg needs meaningful per-dimension variance."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self.layers(representations)


class NormalizedPrototypeLayer(nn.Module):
    """Cosine prototype classifier without weight-normalization state hooks."""

    def __init__(self, input_dim: int, num_prototypes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_prototypes, input_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(inputs, dim=-1), F.normalize(self.weight, dim=-1))


class DINOHead(nn.Module):
    """Map representations to a distribution over learned visual prototypes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        num_prototypes: int,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.prototypes = NormalizedPrototypeLayer(bottleneck_dim, num_prototypes)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self.prototypes(self.projector(representations))


class SelfSupervisedNetwork(nn.Module):
    """A mask-aware encoder followed by an objective-specific training head."""

    def __init__(self, encoder: MaskAwareEncoder, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        representations = self.encoder(images, masks)
        outputs = self.head(representations)
        return {"representations": representations, "outputs": outputs}
