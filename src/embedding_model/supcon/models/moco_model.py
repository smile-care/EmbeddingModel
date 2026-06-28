"""
MoCo 动量对比学习包装层。
包含 query_encoder（可训练）和 momentum_encoder（动量更新，不参与梯度）。
自动根据 backbone_cfg 类型路由到 ConvNeXtModel 或 ViTModel。
"""
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .backbone.dinov3_convnext import DINOv3ConvNextConfig
from .backbone.dinov3_vit import DINOv3ViTConfig
from .convnext_model import ConvNeXtModel
from .vit_model import ViTModel


class MoCoModel(nn.Module):
    """
    MoCo 动量对比学习模型。
    θ_k ← m·θ_k + (1-m)·θ_q
    """

    def __init__(
        self,
        backbone_cfg: Optional[Union[DINOv3ConvNextConfig, DINOv3ViTConfig]] = None,
        ckpt_path: Optional[str] = None,
        embedding_dim: int = 128,
        projection_hidden_dims: List[int] = [256, 128],
        image_size: int = 224,
        freeze_backbone: bool = False,
        fusion_dim: int = 512,
        # ConvNeXt 专用
        use_layers: Optional[List[int]] = [0, 1, 2, 3],
        fpn_out_channels: int = 256,
        seg_layer_idx: int = 0,
        # ViT 专用
        cls_weight: float = 0.3,
        # 共用
        enable_segmentation: bool = True,
        momentum: float = 0.999,
    ):
        """
        Args:
            backbone_cfg: DINOv3ConvNextConfig 或 DINOv3ViTConfig；
                          None 时默认使用 ConvNeXt 配置。
            ckpt_path: 预训练权重路径（.pth）
            embedding_dim: embedding 维度
            projection_hidden_dims: ProjectionHead 隐藏层维度
            image_size: 输入图像边长
            freeze_backbone: 是否冻结 backbone
            fusion_dim: 融合层输出维度（两条管线均有效）
            use_layers: [ConvNeXt] FPN 使用的 stage 索引
            fpn_out_channels: [ConvNeXt] FPN 统一输出通道数
            seg_layer_idx: [ConvNeXt] 分割头使用的 FPN 层索引
            cls_weight: [ViT] CLS token 混合比例
            enable_segmentation: 是否启用分割辅助分支
            momentum: 动量系数
        """
        super().__init__()
        self.momentum = momentum

        is_vit = isinstance(backbone_cfg, DINOv3ViTConfig)

        if is_vit:
            model_cls    = ViTModel
            model_kwargs = dict(
                backbone_cfg=backbone_cfg,
                ckpt_path=ckpt_path,
                embedding_dim=embedding_dim,
                projection_hidden_dims=projection_hidden_dims,
                image_size=image_size,
                freeze_backbone=freeze_backbone,
                fusion_dim=fusion_dim,
                cls_weight=cls_weight,
                enable_segmentation=enable_segmentation,
            )
        else:
            model_cls    = ConvNeXtModel
            model_kwargs = dict(
                backbone_cfg=backbone_cfg,
                ckpt_path=ckpt_path,
                embedding_dim=embedding_dim,
                projection_hidden_dims=projection_hidden_dims,
                image_size=image_size,
                freeze_backbone=freeze_backbone,
                use_layers=use_layers,
                fpn_out_channels=fpn_out_channels,
                fusion_dim=fusion_dim,
                enable_segmentation=enable_segmentation,
                seg_layer_idx=seg_layer_idx,
            )

        self.query_encoder    = model_cls(**model_kwargs)
        self.momentum_encoder = model_cls(**model_kwargs)

        # 初始化 momentum_encoder 与 query_encoder 权重一致
        for p_q, p_k in zip(self.query_encoder.parameters(),
                             self.momentum_encoder.parameters()):
            p_k.data.copy_(p_q.data)

        # momentum_encoder 不参与梯度更新
        for p in self.momentum_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def momentum_update(self) -> None:
        """θ_k ← m·θ_k + (1-m)·θ_q"""
        for p_q, p_k in zip(self.query_encoder.parameters(),
                             self.momentum_encoder.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.data * (1.0 - self.momentum)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mode: str = "query",
        return_features: bool = False,
        return_segmentation: bool = False,
    ) -> dict:
        """
        Args:
            x:    (B, C, H, W)
            mask: (B, 1, H, W)
            mode: "query" 使用可训练编码器；"key" 使用动量编码器
        """
        if mode == "query":
            encoder = self.query_encoder
        elif mode == "key":
            encoder = self.momentum_encoder
            return_segmentation = False
        else:
            raise ValueError(f"mode 必须是 'query' 或 'key'，当前为 {mode!r}")

        return encoder(x, mask, return_features=return_features,
                       return_segmentation=return_segmentation)

    def freeze_backbone_layers(self, num_layers: Optional[int] = None) -> None:
        self.query_encoder.freeze_backbone_layers(num_layers)
        self.momentum_encoder.freeze_backbone_layers(num_layers)

    def unfreeze_all(self) -> None:
        self.query_encoder.unfreeze_all()
        # momentum_encoder 始终不训练
