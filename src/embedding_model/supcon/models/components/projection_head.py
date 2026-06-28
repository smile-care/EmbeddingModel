"""Projection Head for contrastive learning."""
from typing import List

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """MLP projection head.

    Standard design following SimCLR v2 / MoCo v3:
      - Each hidden layer: Linear → BN → ReLU  (no Dropout)
      - Output layer:      Linear only          (no activation, no BN)

    Rationale:
      - No Dropout: contrastive loss is sensitive to embedding L2 norm
        stability; Dropout causes per-sample norm variance that degrades
        cosine similarity computation, especially for MoCo queue embeddings.
      - No activation on the last layer: ReLU would restrict the embedding
        space to the positive orthant, halving the effective cosine space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [384, 384],
        output_dim: int = 128,
    ):
        """
        Args:
            input_dim:   Input dimension (backbone hidden size for ViT,
                         fusion_dim for ConvNeXt).
            hidden_dims: Hidden layer widths. Recommended: match backbone
                         hidden size for ViT (e.g. [384, 384] for ViT-S).
            output_dim:  Final embedding dimension.
        """
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            prev_dim = hidden_dim

        # Output layer: no activation, no BN
        layers.append(nn.Linear(prev_dim, output_dim))

        self.projection = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, output_dim)
        """
        return self.projection(x)
