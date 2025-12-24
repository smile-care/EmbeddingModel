"""
DINOv3 ConvNeXt Backbone模型
支持预训练权重加载和参数冻结
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import DINOv3ConvNextModel


class DINOv3ConvNext(nn.Module):
    """DINOv3 ConvNeXt Backbone模型"""
    
    def __init__(
        self, 
        model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        freeze_backbone: bool = False
    ):
        """
        初始化DINOv3 ConvNeXt模型
        
        Args:
            model_name: 预训练模型名称
            freeze_backbone: 是否冻结backbone参数
        """
        super().__init__()
        
        # 加载预训练模型
        pretrained_model = DINOv3ConvNextModel.from_pretrained(model_name)
        self.stages = pretrained_model.stages
        
        # 冻结参数（如果需要）
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        pixel_values: torch.FloatTensor, 
        output_hidden_states: bool = False
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        前向传播
        
        Args:
            pixel_values: 输入图像 (B, C, H, W)
            output_hidden_states: 是否输出所有层的隐藏状态
            
        Returns:
            如果output_hidden_states=True，返回所有层的特征元组
            否则返回最后一层的特征
        """
        hidden_states = pixel_values

        all_hidden_states = []

        for stage in self.stages:
            hidden_states = stage(hidden_states)

            # 存储中间层输出
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        if output_hidden_states:
            return tuple(all_hidden_states)
        return hidden_states
    
    def freeze(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True