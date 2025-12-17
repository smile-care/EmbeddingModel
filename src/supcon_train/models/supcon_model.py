"""
SupCon模型：DINOv3 Backbone + FPN + Mask特征筛选 + 多层特征融合 + Projection Head
"""
from typing import List, Optional

import torch
import torch.nn as nn

from .backbone.dinov3_convnext import DINOv3ConvNext
from .components import FeatureFusion, FeaturePyramidNetwork, PathAggregationFPN, ProjectionHead


class SupConModel(nn.Module):
    """SupCon模型（使用DINOv3 backbone）"""
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        embedding_dim: int = 128,
        projection_hidden_dims: List[int] = [256, 128],
        image_size: int = 224,
        freeze_backbone: bool = False,
        use_layers: Optional[List[int]] = [1, 2, 3, 4],  # 使用哪些层的特征，如[1, 2, 3, 4]
        fpn_out_channels = 256,  # FPN输出通道数
        fusion_dim: int = 512  # 特征融合后的维度
    ):
        """
        初始化SupCon模型
        
        Args:
            model_name: DINOv3模型名称
            embedding_dim: embedding维度
            projection_hidden_dims: projection head隐藏层维度
            image_size: 输入图像大小
            freeze_backbone: 是否冻结backbone
            use_layers: 使用哪些层的特征（None表示使用所有层）
            fusion_dim: 特征融合后的维度
        """
        super().__init__()
        
        # DINOv3 Backbone
        self.backbone = DINOv3ConvNext(
            model_name=model_name,
            freeze_backbone=freeze_backbone
        )
        
        # 获取backbone各层的特征维度
        # DINOv3 ConvNeXt通常有4个stage
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            dummy_outputs = self.backbone(dummy_input, output_hidden_states=True)
            
            # 计算各层特征维度
            feature_dims = []
            for feat in dummy_outputs:
                if feat.dim() == 4:  # (B, C, H, W)
                    feature_dims.append(feat.shape[1])
                else:
                    feature_dims.append(feat.shape[-1])
        
        # 选择使用的层
        if use_layers is None:
            use_layers = list(range(len(feature_dims)))
        
        # 验证索引有效性
        max_idx = len(feature_dims) - 1
        invalid_indices = [idx for idx in use_layers if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(
                f"use_layers包含无效索引: {invalid_indices}。"
                f"backbone输出{len(feature_dims)}层（索引范围: 0-{max_idx}），"
                f"但配置中使用了: {use_layers}"
            )
        
        self.use_layers = use_layers
        selected_dims = [feature_dims[i] for i in use_layers]
        
        # FPN特征金字塔网络
        # 标准FPN/PNFPN统一所有特征图的通道数为256
        self.fpn = PathAggregationFPN(
            in_channels_list=selected_dims,
            out_channels=fpn_out_channels,  # 统一通道数为256
            num_outs=len(selected_dims),  # 输出与输入相同数量的特征图
            start_level=use_layers[0],
            add_extra_convs=False
        )
        
        # 特征融合模块
        # FPN输出统一通道数，FeatureFusion接收统一通道数的特征
        fpn_output_dims = [fpn_out_channels] * len(selected_dims)
        self.feature_fusion = FeatureFusion(
            feature_dims=fpn_output_dims,  # 使用FPN的统一通道数
            output_dim=fusion_dim
        )
        
        # Projection Head
        self.projection_head = ProjectionHead(
            input_dim=fusion_dim,
            hidden_dims=projection_hidden_dims,
            output_dim=embedding_dim,
            dropout=0.1
        )
        

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_features: bool = False
    ) -> dict:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            mask: mask张量 (B, 1, H, W)
            return_features: 是否返回backbone特征
            
        Returns:
            包含embedding和logits的字典
        """
        # Backbone特征提取（获取所有层）
        all_features = self.backbone(x, output_hidden_states=True)
        
        # 选择使用的层
        selected_features = tuple(all_features[i] for i in self.use_layers)
        
        # FPN特征金字塔网络：优化多尺度特征图
        # FPN统一所有特征图的通道数，通过自顶向下和自底向上路径融合多尺度信息
        fpn_features = self.fpn(selected_features)
        
        # 特征融合（应用mask筛选）
        fused_features = self.feature_fusion(fpn_features, mask)
        
        # Projection
        embeddings = self.projection_head(fused_features)
        
        # 检查embeddings是否包含NaN或Inf
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            print("警告：Projection head输出包含NaN或Inf！")
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        result = {'embeddings': embeddings}
        
        if return_features:
            result['features'] = fused_features
        
        return result
    
    def freeze_backbone_layers(self, num_layers: Optional[int] = None):
        """
        冻结backbone的部分层
        
        Args:
            num_layers: 冻结的层数（None表示冻结全部）
        """
        if num_layers is None:
            self.backbone.freeze()
        else:
            # 冻结前num_layers个stage
            stages = list(self.backbone.stages)
            for i, stage in enumerate(stages[:num_layers]):
                for param in stage.parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有参数"""
        self.backbone.unfreeze()
        for param in self.parameters():
            param.requires_grad = True
