"""
Projection Head模块
"""
from typing import List

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Projection Head：将backbone特征映射到embedding空间"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        初始化Projection Head
        
        Args:
            input_dim: 输入维度（backbone输出维度）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（embedding维度）
            dropout: Dropout比例
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (B, input_dim)
            
        Returns:
            Embedding (B, output_dim)
        """
        return self.projection(x)
