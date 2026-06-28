"""
ViT 管线：DINOv3 ViT Backbone + Mask 加权 Patch Pooling + Projection Head + 分割头
无 FPN，充分利用 ViT 全局注意力实现前景感知 embedding。
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.dinov3_vit import DINOv3ViT, DINOv3ViTConfig
from .components import MaskWeightedPooling, ProjectionHead


class ViTModel(nn.Module):
    """
    ViT 管线模型。

    管线：
        ViT backbone  →  cls_token (B, D)  +  patch_tokens (B, N, D)
        MaskWeightedPooling  →  α·CLS + (1-α)·mask_weighted_patch_pool → (B, D)
        ProjectionHead (3-layer MLP, no Dropout, no activation on last layer)
                             →  (B, embedding_dim)

    并行分割分支：
        patch_tokens reshape → (B, D, nh, nw)
        → ConvTranspose2d 逐步 2× 上采样 → (B, 1, H, W)
    """

    def __init__(
        self,
        backbone_cfg: Optional[DINOv3ViTConfig] = None,
        ckpt_path: Optional[str] = None,
        embedding_dim: int = 128,
        projection_hidden_dims: Optional[List[int]] = None,
        image_size: int = 224,
        freeze_backbone: bool = False,
        fusion_dim: int = 512,   # kept for API compat with MoCoModel, not used in ViT pipeline
        cls_weight: float = 0.3,
        enable_segmentation: bool = True,
    ):
        """
        Args:
            backbone_cfg: DINOv3ViTConfig 实例，None 时使用默认 ViT-S/16 配置
            ckpt_path: 预训练权重路径（.pth，格式为 {"model": state_dict}）
            embedding_dim: 最终 embedding 维度
            projection_hidden_dims: ProjectionHead 隐藏层维度列表。
                None 时自动设为 [D, D]（与 backbone hidden_size 对齐）。
            image_size: 输入图像边长
            freeze_backbone: 是否冻结 backbone 权重
            fusion_dim: 本参数在 ViT 管线中不使用（为与 MoCoModel 接口兼容而保留）
            cls_weight: CLS token 混合比例（0=纯 patch pooling, 1=纯 CLS）
            enable_segmentation: 是否启用分割辅助分支
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

        del fusion_dim  # not used in ViT pipeline; accepted for API compat with MoCoModel
        D = backbone_cfg.hidden_size

        # Mask 加权 Pooling：纯 pooling，输出 (B, D)，不做维度变换
        self.mask_pooling = MaskWeightedPooling(cls_weight=cls_weight)

        # ProjectionHead：D → D → D → embedding_dim（3-layer MLP，无 Dropout）
        if projection_hidden_dims is None:
            projection_hidden_dims = [D, D]

        self.projection_head = ProjectionHead(
            input_dim=D,
            hidden_dims=projection_hidden_dims,
            output_dim=embedding_dim,
        )

        # 分割头：patch tokens → 空间特征图 → 逐步上采样至原图
        self.enable_segmentation = enable_segmentation
        if enable_segmentation:
            patch_size     = backbone_cfg.patch_size
            upsample_steps = int(torch.tensor(patch_size).float().log2().item())  # log2(16)=4
            layers: List[nn.Module] = []
            in_ch = D
            for _ in range(upsample_steps):
                out_ch = max(32, in_ch // 2)
                layers += [
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
                in_ch = out_ch
            layers += [nn.Conv2d(in_ch, 1, kernel_size=1), nn.Sigmoid()]
            self.segmentation_head = nn.Sequential(*layers)
            self._init_seg_head()

    def _init_seg_head(self) -> None:
        for m in self.segmentation_head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
            return_features: 是否返回 pooling 后的特征向量
            return_segmentation: 未使用，分割结果由 enable_segmentation 控制

        Returns:
            dict:
                embeddings:   (B, embedding_dim)
                features:     (B, fusion_dim)        — 仅 return_features=True
                segmentation: (B, 1, H, W) in [0,1] — 仅 enable_segmentation=True
        """
        cls_token, patch_tokens = self.backbone(x, output_hidden_states=True)
        fused      = self.mask_pooling(cls_token, patch_tokens, mask)
        embeddings = self.projection_head(fused)

        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

        result = {"embeddings": embeddings}

        if return_features:
            result["features"] = fused

        if self.enable_segmentation:
            B, N, D = patch_tokens.shape
            nh = nw  = int(N ** 0.5)
            seg_feat   = patch_tokens.transpose(1, 2).view(B, D, nh, nw)
            seg_logits = self.segmentation_head(seg_feat)
            if seg_logits.shape[2:] != mask.shape[2:]:
                seg_logits = F.interpolate(
                    seg_logits, size=mask.shape[2:], mode="bilinear", align_corners=False
                )
            result["segmentation"] = seg_logits

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
