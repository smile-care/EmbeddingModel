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
        image_size: int = 224,
        freeze_backbone: bool = False,
        # ConvNeXt 专用
        use_layers: Optional[List[int]] = [1, 2, 3],
        # ViT 专用
        cls_weight: float = 0.3,
        momentum: float = 0.999,
    ):
        """
        Args:
            backbone_cfg: DINOv3ConvNextConfig 或 DINOv3ViTConfig；
                          None 时默认使用 ConvNeXt 配置。
            ckpt_path: 预训练权重路径（.pth）
            embedding_dim: embedding 维度
            image_size: 输入图像边长
            freeze_backbone: 是否冻结 backbone
            use_layers: [ConvNeXt] 原生多层融合使用的 stage 索引
            cls_weight: [ViT] CLS token 混合比例
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
                image_size=image_size,
                freeze_backbone=freeze_backbone,
                cls_weight=cls_weight,
            )
        else:
            model_cls    = ConvNeXtModel
            model_kwargs = dict(
                backbone_cfg=backbone_cfg,
                ckpt_path=ckpt_path,
                embedding_dim=embedding_dim,
                image_size=image_size,
                freeze_backbone=freeze_backbone,
                use_layers=use_layers,
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
        else:
            raise ValueError(f"mode 必须是 'query' 或 'key'，当前为 {mode!r}")

        return encoder(x, mask, return_features=return_features)

    def freeze_backbone_layers(self, num_layers: Optional[int] = None) -> None:
        self.query_encoder.freeze_backbone_layers(num_layers)
        self.momentum_encoder.freeze_backbone_layers(num_layers)

    def unfreeze_all(self) -> None:
        self.query_encoder.unfreeze_all()
        # momentum_encoder 始终不训练
