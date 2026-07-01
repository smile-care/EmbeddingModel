"""
多层特征融合模块
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """Backbone 原生多层特征的 mask 加权融合模块。"""
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int = 512,
        use_layers: List[int] = [1, 2, 3]
    ):
        """
        初始化特征融合模块
        
        Args:
            feature_dims: backbone 各 stage 特征的通道数
            output_dim: 输出特征维度
            use_layers: 使用哪些层的特征进行融合
        """
        super().__init__()
        
        self.use_layers = list(use_layers)
        if not self.use_layers:
            raise ValueError("use_layers 不能为空")
        invalid_layers = [i for i in self.use_layers if i < 0 or i >= len(feature_dims)]
        if invalid_layers:
            raise ValueError(
                f"use_layers 包含无效 stage: {invalid_layers}, feature_dims 长度为 {len(feature_dims)}"
            )
        self.feature_dims = [feature_dims[i] for i in self.use_layers]
        
        # 只为 use_layers 指定的 stage 创建投影层，避免留下未参与 forward 的参数。
        # 不使用 AdaptiveAvgPool2d，空间聚合由 mask 加权池化完成。
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            ) for dim in self.feature_dims
        ])
        
        # 融合层
        # 注意：不使用 Dropout——对比学习对 embedding 范数稳定性敏感，
        # Dropout 会导致 per-sample 范数方差，破坏余弦相似度计算；
        # 且 MoCo 模式下 query/key 编码器的 Dropout 状态不一致会引入噪声梯度。
        # 已有 LayerNorm + AdamW weight_decay + 数据增强提供充分正则化。
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(self.use_layers), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _masked_pool(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask-weighted average pooling with global pooling fallback for empty masks."""
        B, C, H, W = feat.shape
        mask_resized = F.interpolate(mask.float(), size=(H, W), mode="area")
        mask_weights = torch.clamp(mask_resized, 0.0, 1.0)

        weighted_sum = (feat * mask_weights).sum(dim=(2, 3))
        weight_sum = mask_weights.sum(dim=(2, 3))
        masked_pool = weighted_sum / weight_sum.clamp_min(1e-8)

        global_pool = feat.mean(dim=(2, 3))
        has_foreground = (weight_sum > 1e-6).expand_as(masked_pool)
        return torch.where(has_foreground, masked_pool, global_pool)

    def _validate_inputs(self, features: Tuple[torch.Tensor, ...], mask: torch.Tensor) -> None:
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"mask 必须是 (B, 1, H, W)，当前 shape={tuple(mask.shape)}")
        if len(features) <= max(self.use_layers):
            raise ValueError(
                f"features 长度不足：需要 stage {max(self.use_layers)}，当前长度 {len(features)}"
            )
        for i in self.use_layers:
            feat = features[i]
            if feat.ndim != 4:
                raise ValueError(f"features[{i}] 必须是 (B, C, H, W)，当前 shape={tuple(feat.shape)}")
            if feat.shape[0] != mask.shape[0]:
                raise ValueError(
                    f"features[{i}] batch size ({feat.shape[0]}) 与 mask ({mask.shape[0]}) 不一致"
                )
    
    def forward(self, features: Tuple[torch.Tensor, ...], mask: torch.Tensor) -> torch.Tensor:
        """
        融合多层特征
        
        Args:
            features: 多层特征元组，每层形状为 (B, C, H, W)
            mask: mask张量 (B, 1, H, W)，用于特征筛选
            
        Returns:
            融合后的特征 (B, output_dim)
        """
        self._validate_inputs(features, mask)
        projected_features = []
        
        for proj_idx, i in enumerate(self.use_layers):
            feat = features[i]
            pooled_feat = self._masked_pool(feat, mask)
            proj_feat = self.projections[proj_idx](pooled_feat)  # (B, output_dim)
            projected_features.append(proj_feat)
        
        # 拼接所有层特征
        fused = torch.cat(projected_features, dim=1)
        
        # 融合
        output = self.fusion(fused)
        
        return output
