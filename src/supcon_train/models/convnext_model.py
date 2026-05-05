"""
ConvNeXt 管线：DINOv3 ConvNeXt Backbone + PA-FPN + Mask 加权融合 + Projection Head + 分割头
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.dinov3_convnext import DINOv3ConvNext, DINOv3ConvNextConfig
from .components import FeatureFusion, PathAggregationFPN, ProjectionHead


class ConvNeXtModel(nn.Module):
    """
    ConvNeXt 管线模型。

    管线：
        ConvNeXt backbone  →  (B, C0, H/4, W/4) ... (B, C3, H/32, W/32)
        PA-FPN             →  统一通道数，自顶向下 + 自底向上多尺度融合
        FeatureFusion      →  mask 加权池化，拼接各层后线性融合 → (B, fusion_dim)
        ProjectionHead     →  (B, embedding_dim)

    并行分割分支：
        FPN 指定层特征 → 轻量卷积头 → (B, 1, H, W)
    """

    def __init__(
        self,
        backbone_cfg: Optional[DINOv3ConvNextConfig] = None,
        ckpt_path: Optional[str] = None,
        embedding_dim: int = 128,
        projection_hidden_dims: List[int] = [256, 128],
        image_size: int = 224,
        freeze_backbone: bool = False,
        use_layers: Optional[List[int]] = [0, 1, 2, 3],
        fpn_out_channels: int = 256,
        fusion_dim: int = 512,
        enable_segmentation: bool = True,
        seg_layer_idx: int = 0,
    ):
        """
        Args:
            backbone_cfg: DINOv3ConvNextConfig 实例，None 时使用默认 tiny 配置
            ckpt_path: 预训练权重路径（.pth，格式为 {"model": state_dict}）
            embedding_dim: 最终 embedding 维度
            projection_hidden_dims: ProjectionHead 隐藏层维度列表
            image_size: 输入图像边长
            freeze_backbone: 是否冻结 backbone 权重
            use_layers: 使用 FPN 哪些层（0=4x, 1=8x, 2=16x, 3=32x）
            fpn_out_channels: FPN 统一输出通道数
            fusion_dim: FeatureFusion 输出维度
            enable_segmentation: 是否启用分割辅助分支
            seg_layer_idx: 分割头使用的 FPN 层索引（0 为最高分辨率）
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

        # PA-FPN
        self.fpn = PathAggregationFPN(
            in_channels_list=feature_dims,
            out_channels=fpn_out_channels,
            num_outs=len(feature_dims),
            start_level=use_layers[0],
            add_extra_convs=False,
        )

        # Mask 加权融合
        self.feature_fusion = FeatureFusion(
            feature_dims=[fpn_out_channels] * len(use_layers),
            output_dim=fusion_dim,
            use_layers=use_layers,
        )

        # Projection Head
        self.projection_head = ProjectionHead(
            input_dim=fusion_dim,
            hidden_dims=projection_hidden_dims,
            output_dim=embedding_dim,
            dropout=0.1,
        )

        # 分割头
        self.enable_segmentation = enable_segmentation
        self.seg_layer_idx = seg_layer_idx
        if enable_segmentation:
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(fpn_out_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            self._init_seg_head()

    def _init_seg_head(self) -> None:
        for m in self.segmentation_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_features: bool = False,
        return_segmentation: bool = False,
    ) -> dict:
        """
        Args:
            x:    (B, C, H, W) 输入图像
            mask: (B, 1, H, W) 前景 mask，值域 [0, 1]
            return_features: 是否返回融合后的特征向量
            return_segmentation: 未使用，分割结果由 enable_segmentation 控制

        Returns:
            dict:
                embeddings:   (B, embedding_dim)
                features:     (B, fusion_dim)        — 仅 return_features=True
                segmentation: (B, 1, H, W) in [0,1] — 仅 enable_segmentation=True
        """
        features     = self.backbone(x, output_hidden_states=True)
        fpn_features = self.fpn(features)
        fused        = self.feature_fusion(fpn_features, mask)
        embeddings   = self.projection_head(fused)

        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

        result = {"embeddings": embeddings}

        if return_features:
            result["features"] = fused

        if self.enable_segmentation:
            seg_feat   = fpn_features[self.seg_layer_idx]
            seg_logits = self.segmentation_head(seg_feat)
            if seg_logits.shape[2:] != mask.shape[2:]:
                seg_logits = F.interpolate(
                    seg_logits, size=mask.shape[2:], mode="bilinear", align_corners=False
                )
            result["segmentation"] = seg_logits

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
