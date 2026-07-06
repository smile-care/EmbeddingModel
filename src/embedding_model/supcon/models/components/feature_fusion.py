"""ConvNeXt multi-stage foreground/context representation fusion."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """Mask-aware multi-stage representation neck.

    Supported modes:
      - fg_only: pool each selected stage inside the instance mask.
      - fg_context_delta: pool foreground, nearby context ring, and their
        difference, then project the concatenated vector.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int = 512,
        use_layers: List[int] = [1, 2, 3],
        pooling_mode: str = "fg_only",
        pooling_config: Optional[dict] = None,
    ):
        """
        初始化特征融合模块
        
        Args:
            feature_dims: backbone 各 stage 特征的通道数
            output_dim: 输出特征维度
            use_layers: 使用哪些层的特征进行融合
        """
        super().__init__()
        
        valid_modes = {"fg_only", "fg_context_delta"}
        if pooling_mode not in valid_modes:
            raise ValueError(f"FeatureFusion pooling_mode 必须是 {sorted(valid_modes)}，当前为 {pooling_mode!r}")

        self.use_layers = list(use_layers)
        self.pooling_mode = pooling_mode
        pooling_config = pooling_config or {}
        self.context_dilation = int(pooling_config.get("context_dilation", 1))
        if self.context_dilation < 1:
            raise ValueError(f"context_dilation 必须 >= 1，当前为 {self.context_dilation}")
        if not self.use_layers:
            raise ValueError("use_layers 不能为空")
        invalid_layers = [i for i in self.use_layers if i < 0 or i >= len(feature_dims)]
        if invalid_layers:
            raise ValueError(
                f"use_layers 包含无效 stage: {invalid_layers}, feature_dims 长度为 {len(feature_dims)}"
            )
        self.feature_dims = [feature_dims[i] for i in self.use_layers]
        pooled_dims = [
            dim * 3 if self.pooling_mode == "fg_context_delta" else dim
            for dim in self.feature_dims
        ]

        stage_gate_cfg = pooling_config.get("stage_gate", {})
        stage_gate_cfg = stage_gate_cfg if isinstance(stage_gate_cfg, dict) else {}
        self.stage_gate_enabled = bool(stage_gate_cfg.get("enabled", False))
        if self.stage_gate_enabled:
            self.stage_logits = nn.Parameter(torch.zeros(len(self.use_layers)))
        else:
            self.register_parameter("stage_logits", None)
        
        # 只为 use_layers 指定的 stage 创建投影层，避免留下未参与 forward 的参数。
        # 不使用 AdaptiveAvgPool2d，空间聚合由 mask 加权池化完成。
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            ) for dim in pooled_dims
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

    def _context_ring(self, mask_weights: torch.Tensor) -> torch.Tensor:
        """Build a local background ring around the foreground on a feature map."""
        kernel_size = 2 * self.context_dilation + 1
        dilated = F.max_pool2d(
            mask_weights,
            kernel_size=kernel_size,
            stride=1,
            padding=self.context_dilation,
        )
        return torch.clamp(dilated - mask_weights, 0.0, 1.0)

    def _pool_stage(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "fg_only":
            return self._masked_pool(feat, mask)

        mask_resized = F.interpolate(mask.float(), size=feat.shape[-2:], mode="area")
        fg_mask = torch.clamp(mask_resized, 0.0, 1.0)
        ctx_mask = self._context_ring(fg_mask)

        fg_pool = self._masked_pool(feat, fg_mask)
        ctx_pool = self._masked_pool(feat, ctx_mask)
        return torch.cat([fg_pool, ctx_pool, fg_pool - ctx_pool], dim=1)

    def get_stage_weights(self) -> Optional[torch.Tensor]:
        """Return normalized stage gate weights when stage gating is enabled."""
        if not self.stage_gate_enabled or self.stage_logits is None:
            return None
        return torch.softmax(self.stage_logits.detach(), dim=0)

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
            pooled_feat = self._pool_stage(feat, mask)
            proj_feat = self.projections[proj_idx](pooled_feat)  # (B, output_dim)
            projected_features.append(proj_feat)

        if self.stage_gate_enabled and self.stage_logits is not None:
            stage_weights = torch.softmax(self.stage_logits, dim=0)
            projected_features = [
                stage_weights[i] * feat
                for i, feat in enumerate(projected_features)
            ]
        
        # 拼接所有层特征
        fused = torch.cat(projected_features, dim=1)
        
        # 融合
        output = self.fusion(fused)
        
        return output
