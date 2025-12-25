"""
多层特征融合模块
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """多层特征融合模块"""
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int = 512,
        use_layers: List[int] = [0, 1, 2]
    ):
        """
        初始化特征融合模块
        
        Args:
            feature_dims: 各层特征的维度列表
            output_dim: 输出特征维度
            use_layers: 使用哪些层的特征进行融合
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.num_layers = len(feature_dims)
        self.use_layers = use_layers
        
        # 为每层特征创建投影层
        # 注意：这里不使用AdaptiveAvgPool2d，而是在forward中使用mask加权池化
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * self.num_layers, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
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
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: Tuple[torch.Tensor, ...], mask: torch.Tensor) -> torch.Tensor:
        """
        融合多层特征
        
        Args:
            features: 多层特征元组，每层形状为 (B, C, H, W)
            mask: mask张量 (B, 1, H, W)，用于特征筛选
            
        Returns:
            融合后的特征 (B, output_dim)
        """
        # 对每层特征应用mask加权和投影
        projected_features = []
        
        for i, feat in enumerate(features):
            if i not in self.use_layers:
                continue
            # Resize mask到特征图尺寸
            B, C, H, W = feat.shape
            mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')  # (B, 1, H, W)
            
            mask_weights = torch.clamp(mask_resized, 0.02, 1.0)  # (B, 1, H, W)
            
            # 方法：对每个通道，使用mask权重进行加权平均
            weighted_feat = feat * mask_weights  # (B, C, H, W)
            
            # 计算mask加权后的全局平均池化
            # mask_sum: (B, 1)，每个样本的mask权重总和
            mask_sum = mask_weights.sum(dim=(2, 3)) + 1e-8  # (B, 1)
            # 加权平均：对空间维度求和后除以mask权重总和
            # weighted_feat.sum(dim=(2, 3)): (B, C)
            # mask_sum: (B, 1)，广播除法得到 (B, C)
            pooled_feat = weighted_feat.sum(dim=(2, 3)) / mask_sum  # (B, C)
            
            # 投影到统一维度
            proj_feat = self.projections[i](pooled_feat)  # (B, output_dim)
            projected_features.append(proj_feat)
        
        # 拼接所有层特征
        fused = torch.cat(projected_features, dim=1)
        
        # 融合
        output = self.fusion(fused)
        
        return output
