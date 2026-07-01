"""
ConvNeXt 管线：DINOv3 ConvNeXt Backbone + 原生多层 Mask 加权融合 + Projection Head
"""
from typing import List, Optional

import torch
import torch.nn as nn

from .backbone.dinov3_convnext import DINOv3ConvNext, DINOv3ConvNextConfig
from .components import FeatureFusion, ProjectionHead


class ConvNeXtModel(nn.Module):
    """
    ConvNeXt 管线模型。

    管线：
        ConvNeXt backbone  →  (B, C0, H/4, W/4) ... (B, C3, H/32, W/32)
        FeatureFusion      →  原生多层 mask 加权池化，拼接后线性融合
        ProjectionHead     →  (B, embedding_dim)

    """

    def __init__(
        self,
        backbone_cfg: Optional[DINOv3ConvNextConfig] = None,
        ckpt_path: Optional[str] = None,
        embedding_dim: int = 128,
        image_size: int = 224,
        freeze_backbone: bool = False,
        use_layers: Optional[List[int]] = [1, 2, 3],
    ):
        """
        Args:
            backbone_cfg: DINOv3ConvNextConfig 实例，None 时使用默认 tiny 配置
            ckpt_path: 预训练权重路径（.pth，格式为 {"model": state_dict}）
            embedding_dim: 最终 embedding 维度
            image_size: 输入图像边长
            freeze_backbone: 是否冻结 backbone 权重
            use_layers: 使用 backbone 哪些 stage（0=4x, 1=8x, 2=16x, 3=32x）
        """
        super().__init__()

        self.image_size = image_size
        self.use_layers = use_layers
        # Backbone
        self.backbone = DINOv3ConvNext(
            cfg=backbone_cfg,
            ckpt_path=ckpt_path,
            freeze_backbone=freeze_backbone,
        )

        # 推断各 stage 输出通道数
        with torch.no_grad():
            _dummy = torch.randn(1, 3, image_size, image_size)
            _feats = self.backbone(_dummy, output_hidden_states=True)
            feature_dims = [f.shape[1] for f in _feats]  # (B, C, H, W)

        # 原生多层特征 mask 加权融合
        self.feature_fusion = FeatureFusion(
            feature_dims=feature_dims,
            output_dim=embedding_dim,
            use_layers=use_layers,
        )

        # Projection Head
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
        """
        Args:
            x:    (B, C, H, W) 输入图像
            mask: (B, 1, H, W) 前景 mask，值域 [0, 1]
            return_features: 是否返回融合后的特征向量

        Returns:
            dict:
                embeddings:   (B, embedding_dim)
                features:     FeatureFusion 输出向量 — 仅 return_features=True
        """
        features     = self.backbone(x, output_hidden_states=True)
        fused        = self.feature_fusion(features, mask)
        embeddings   = self.projection_head(fused)

        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

        result = {"embeddings": embeddings}

        if return_features:
            result["features"] = fused

        return result

    def freeze_backbone_layers(self, num_layers: Optional[int] = None) -> None:
        """冻结前 num_layers 个 stage（None 表示全部冻结）。"""
        if num_layers is None:
            self.backbone.freeze()
        else:
            for stage in list(self.backbone.stages)[:num_layers]:
                for param in stage.parameters():
                    param.requires_grad = False

    def unfreeze_all(self) -> None:
        self.backbone.unfreeze()
        for param in self.parameters():
            param.requires_grad = True
