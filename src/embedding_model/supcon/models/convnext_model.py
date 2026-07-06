"""
ConvNeXt 管线：DINOv3 ConvNeXt Backbone + 原生多层 Mask 加权融合 + Projection Head
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        fusion_dim: Optional[int] = None,
        projection_hidden_dim: Optional[int] = None,
        image_size: int = 224,
        freeze_backbone: bool = False,
        use_layers: Optional[List[int]] = [1, 2, 3],
        mask_gating: Optional[dict] = None,
        pooling: Optional[dict] = None,
    ):
        """
        Args:
            backbone_cfg: DINOv3ConvNextConfig 实例，None 时使用默认 tiny 配置
            ckpt_path: 预训练权重路径（.pth，格式为 {"model": state_dict}）
            embedding_dim: ProjectionHead 输出维度（对比学习监督空间 z）
            fusion_dim: FeatureFusion 输出维度（应用分析表征空间 h）
            projection_hidden_dim: ProjectionHead 隐藏层维度
            image_size: 输入图像边长
            freeze_backbone: 是否冻结 backbone 权重
            use_layers: 使用 backbone 哪些 stage（0=4x, 1=8x, 2=16x, 3=32x）
        """
        super().__init__()

        self.image_size = image_size
        self.use_layers = list(use_layers or [1, 2, 3])
        self.fusion_dim = int(fusion_dim or embedding_dim)
        self.embedding_dim = int(embedding_dim)

        mask_gating = mask_gating or {}
        self.mask_gating_enabled = bool(mask_gating.get("enabled", False))
        init_gamma = float(mask_gating.get("init_gamma", 0.0))
        if self.mask_gating_enabled:
            self.mask_gammas = nn.Parameter(torch.full((len(self.use_layers),), init_gamma))
        else:
            self.register_parameter("mask_gammas", None)

        pooling = pooling or {}
        self.pooling_mode = str(pooling.get("mode", "fg_only"))
        self.pooling_config = dict(pooling)
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
            output_dim=self.fusion_dim,
            use_layers=self.use_layers,
            pooling_mode=self.pooling_mode,
            pooling_config=self.pooling_config,
        )

        # Projection Head
        self.projection_head = ProjectionHead(
            input_dim=self.fusion_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=self.embedding_dim,
        )

    def _apply_mask_gating(
        self,
        features: tuple[torch.Tensor, ...],
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if not self.mask_gating_enabled or self.mask_gammas is None:
            return features

        gated = list(features)
        for gamma_idx, layer_idx in enumerate(self.use_layers):
            feat = gated[layer_idx]
            mask_resized = F.interpolate(mask.float(), size=feat.shape[-2:], mode="area")
            mask_weights = torch.clamp(mask_resized, 0.0, 1.0)
            gated[layer_idx] = feat * (1.0 + self.mask_gammas[gamma_idx] * mask_weights)
        return tuple(gated)

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
            return_features: 保留旧调用参数；输出始终包含 representations/projections

        Returns:
            dict:
                representations: FeatureFusion 输出表征 h，供应用分析使用
                projections:     ProjectionHead 输出投影 z，供 SupCon/MoCo loss 使用
        """
        features = self.backbone(x, output_hidden_states=True)
        features = self._apply_mask_gating(features, mask)
        representations = self.feature_fusion(features, mask)
        projections = self.projection_head(representations)

        if torch.isnan(projections).any() or torch.isinf(projections).any():
            projections = torch.nan_to_num(projections, nan=0.0, posinf=1.0, neginf=-1.0)

        return {"representations": representations, "projections": projections}

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
