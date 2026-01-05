"""
MoCo模型：基于SupConModel的动量对比学习模型
包含query_encoder（可训练）和momentum_encoder（动量更新）
"""
from typing import List, Optional

import torch
import torch.nn as nn

from .supcon_model import SupConModel


class MoCoModel(nn.Module):
    """
    MoCo模型：使用动量更新的编码器进行对比学习
    
    核心思想：
    - query_encoder: 当前训练的编码器（可训练，通过梯度更新）
    - momentum_encoder: 动量更新的编码器（不训练，通过动量更新）
    - 动量更新：θ_k ← m·θ_k + (1-m)·θ_q
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        embedding_dim: int = 128,
        projection_hidden_dims: List[int] = [256, 128],
        image_size: int = 448,
        freeze_backbone: bool = False,
        use_layers: Optional[List[int]] = [0, 1, 2],
        fpn_out_channels: int = 256,
        fusion_dim: int = 512,
        momentum: float = 0.999,
        enable_segmentation: bool = True,
        seg_layer_idx: int = 0
    ):
        """
        初始化MoCo模型
        
        Args:
            model_name: DINOv3模型名称
            embedding_dim: embedding维度
            projection_hidden_dims: projection head隐藏层维度
            image_size: 输入图像大小
            freeze_backbone: 是否冻结backbone
            use_layers: 使用FPN的哪些层的特征
            fpn_out_channels: FPN输出通道数
            fusion_dim: 特征融合后的维度
            momentum: 动量系数（默认0.999）
            enable_segmentation: 是否启用语义分割分支
            seg_layer_idx: 用于分割的FPN层索引
        """
        super().__init__()
        
        self.momentum = momentum
        
        # Query encoder: 可训练的编码器
        self.query_encoder = SupConModel(
            model_name=model_name,
            embedding_dim=embedding_dim,
            projection_hidden_dims=projection_hidden_dims,
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            use_layers=use_layers,
            fpn_out_channels=fpn_out_channels,
            fusion_dim=fusion_dim,
            enable_segmentation=enable_segmentation,
            seg_layer_idx=seg_layer_idx
        )
        
        # Momentum encoder: 动量更新的编码器（不训练）
        self.momentum_encoder = SupConModel(
            model_name=model_name,
            embedding_dim=embedding_dim,
            projection_hidden_dims=projection_hidden_dims,
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            use_layers=use_layers,
            fpn_out_channels=fpn_out_channels,
            fusion_dim=fusion_dim,
            enable_segmentation=enable_segmentation,
            seg_layer_idx=seg_layer_idx
        )
        
        # 初始化momentum_encoder的参数与query_encoder相同
        self._init_momentum_encoder()
        
        # 冻结momentum_encoder的参数（不通过梯度更新）
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
    
    def _init_momentum_encoder(self):
        """初始化momentum_encoder的参数，使其与query_encoder相同"""
        for param_q, param_k in zip(
            self.query_encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)
    
    @torch.no_grad()
    def momentum_update(self):
        """
        动量更新momentum_encoder的参数
        公式：θ_k ← m·θ_k + (1-m)·θ_q
        """
        for param_q, param_k in zip(
            self.query_encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mode: str = 'query',
        return_features: bool = False,
        return_segmentation: bool = False
    ) -> dict:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            mask: mask张量 (B, 1, H, W)
            mode: 'query' 或 'key'
                - 'query': 使用query_encoder（可训练）
                - 'key': 使用momentum_encoder（不训练，用于生成队列中的负样本）
            return_features: 是否返回backbone特征
            return_segmentation: 是否返回分割结果（仅对query模式有效）
            
        Returns:
            包含embedding和可选features/segmentation的字典
        """
        if mode == 'query':
            encoder = self.query_encoder
        elif mode == 'key':
            encoder = self.momentum_encoder
            # key模式通常不需要分割结果（用于生成队列中的负样本）
            return_segmentation = False
        else:
            raise ValueError(f"mode必须是'query'或'key'，当前为{mode}")
        
        # 使用对应的编码器进行前向传播
        return encoder(x, mask, return_features=return_features, return_segmentation=return_segmentation)
    
    def freeze_backbone_layers(self, num_layers: Optional[int] = None):
        """
        冻结backbone的部分层（同时作用于query_encoder和momentum_encoder）
        
        Args:
            num_layers: 冻结的层数（None表示冻结全部）
        """
        self.query_encoder.freeze_backbone_layers(num_layers)
        self.momentum_encoder.freeze_backbone_layers(num_layers)
    
    def unfreeze_all(self):
        """解冻所有参数（仅作用于query_encoder，momentum_encoder始终不训练）"""
        self.query_encoder.unfreeze_all()
        # momentum_encoder的参数保持冻结状态

