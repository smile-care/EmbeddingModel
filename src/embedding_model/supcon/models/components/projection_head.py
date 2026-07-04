"""Projection Head for contrastive learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head.

    Batch-size independent projection:
      - Hidden layer: Linear → LayerNorm → GELU  (no Dropout)
      - Output layer: Linear → L2Norm

    Rationale:
      - No Dropout: contrastive loss is sensitive to embedding L2 norm
        stability; Dropout causes per-sample norm variance that degrades
        cosine similarity computation, especially for MoCo queue embeddings.
      - No activation on the last layer: ReLU would restrict the embedding
        space to the positive orthant, halving the effective cosine space.
      - LayerNorm avoids BatchNorm's train/eval and small-batch statistic
        mismatch, which is important for fast web fine-tuning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int = 128,
    ):
        """
        Args:
            input_dim:   Input dimension (backbone hidden size for ViT,
                         multi-stage fusion output for ConvNeXt).
            output_dim:  Final embedding dimension.
        """
        super().__init__()

        hidden_dim = hidden_dim or max(input_dim, output_dim * 4)
        prev_dim = input_dim

        self.projection = nn.Sequential(
            nn.Linear(prev_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            # Output layer: no activation; final L2 normalization is in forward.
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, output_dim)
        """
        return F.normalize(self.projection(x), dim=1, p=2, eps=1e-8)
