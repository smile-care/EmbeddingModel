"""
ViT 管线：DINOv3 ViT Backbone + Mask 加权 Patch Pooling + Projection Head
充分利用 ViT 全局注意力实现前景感知 embedding。
"""
from typing import Optional

import torch
import torch.nn as nn

from .backbone.dinov3_vit import DINOv3ViT, DINOv3ViTConfig
from .components import MaskWeightedPooling, ProjectionHead


class ViTModel(nn.Module):
    """
    ViT 管线模型。

    管线：
        ViT backbone  →  cls_token (B, D)  +  patch_tokens (B, N, D)
        MaskWeightedPooling  →  α·CLS + (1-α)·mask_weighted_patch_pool → (B, D)
        ProjectionHead     →  Linear + LayerNorm + GELU + Linear + L2Norm
                             →  (B, embedding_dim)

    """

    def __init__(
        self,
        backbone_cfg: Optional[DINOv3ViTConfig] = None,
        ckpt_path: Optional[str] = None,
        embedding_dim: int = 128,
        image_size: int = 224,
        freeze_backbone: bool = False,
        cls_weight: float = 0.3,
    ):
        """
        Args:
            backbone_cfg: DINOv3ViTConfig 实例，None 时使用默认 ViT-S/16 配置
            ckpt_path: 预训练权重路径（.pth，格式为 {"model": state_dict}）
            embedding_dim: 最终 embedding 维度
            image_size: 输入图像边长
            freeze_backbone: 是否冻结 backbone 权重
            cls_weight: CLS token 混合比例（0=纯 patch pooling, 1=纯 CLS）
        """
        super().__init__()

        if backbone_cfg is None:
            backbone_cfg = DINOv3ViTConfig()

        self.backbone_cfg = backbone_cfg
        self.image_size   = image_size

        # Backbone
        self.backbone = DINOv3ViT(
            cfg=backbone_cfg,
            ckpt_path=ckpt_path,
            freeze_backbone=freeze_backbone,
        )

        D = backbone_cfg.hidden_size

        # Mask 加权 Pooling：纯 pooling，输出 (B, D)，不做维度变换
        self.mask_pooling = MaskWeightedPooling(cls_weight=cls_weight)

        # ProjectionHead: D → 4 * embedding_dim → embedding_dim
        self.projection_head = ProjectionHead(
            input_dim=D,
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
            return_features: 是否返回 pooling 后的特征向量

        Returns:
            dict:
                embeddings:   (B, embedding_dim)
                features:     pooling 后的特征向量 — 仅 return_features=True
        """
        cls_token, patch_tokens = self.backbone(x, output_hidden_states=True)
        fused      = self.mask_pooling(cls_token, patch_tokens, mask)
        embeddings = self.projection_head(fused)

        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

        result = {"embeddings": embeddings}

        if return_features:
            result["features"] = fused

        return result

    def freeze_backbone_layers(self, num_layers: Optional[int] = None) -> None:
        """冻结前 num_layers 个 transformer block（None 表示全部冻结）。"""
        if num_layers is None:
            self.backbone.freeze()
        else:
            for block in list(self.backbone.model["layer"])[:num_layers]:
                for param in block.parameters():
                    param.requires_grad = False

    def unfreeze_all(self) -> None:
        self.backbone.unfreeze()
        for param in self.parameters():
            param.requires_grad = True
