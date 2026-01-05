"""
SupCon模型：DINOv3 Backbone + FPN + Mask特征筛选 + 多层特征融合 + Projection Head + 语义分割分支
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
        image_size: int = 448,
        freeze_backbone: bool = False,
        use_layers: Optional[List[int]] = [0, 1, 2],  # 使用FPN的哪些层的特征(FPN共有4层，索引0-3, 4x, 8x, 16x, 32x)
        fpn_out_channels = 256,  # FPN输出通道数
        fusion_dim: int = 512,  # 特征融合后的维度
        enable_segmentation: bool = True,  # 是否启用语义分割分支
        seg_layer_idx: int = 0  # 用于分割的FPN层索引（0为最高分辨率层）
    ):
        """
        初始化SupCon模型
        
        Args:
            model_name: DINOv3模型名称
            embedding_dim: embedding维度
            projection_hidden_dims: projection head隐藏层维度
            image_size: 输入图像大小
            freeze_backbone: 是否冻结backbone
            use_layers: 使用FPN的哪些层的特征（None表示使用所有层）
            fusion_dim: 特征融合后的维度
            enable_segmentation: 是否启用语义分割分支
            seg_layer_idx: 用于分割的FPN层索引（0为最高分辨率层）
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
        
        self.use_layers = use_layers
        self.image_size = image_size
        
        # FPN特征金字塔网络
        # 标准FPN/PNFPN统一所有特征图的通道数为256
        self.fpn = PathAggregationFPN(
            in_channels_list=feature_dims,
            out_channels=fpn_out_channels,  # 统一通道数为256
            num_outs=len(feature_dims),  # 输出与输入相同数量的特征图
            start_level=use_layers[0],
            add_extra_convs=False
        )
        
        # 特征融合模块
        # FPN输出统一通道数，FeatureFusion接收统一通道数的特征
        fpn_output_dims = [fpn_out_channels] * len(use_layers)
        self.feature_fusion = FeatureFusion(
            feature_dims=fpn_output_dims,  # 使用FPN的统一通道数
            output_dim=fusion_dim,
            use_layers=use_layers
        )
        
        # Projection Head
        self.projection_head = ProjectionHead(
            input_dim=fusion_dim,
            hidden_dims=projection_hidden_dims,
            output_dim=embedding_dim,
            dropout=0.1
        )
        
        # 语义分割分支（与对比学习分支并行）
        self.enable_segmentation = enable_segmentation
        self.seg_layer_idx = seg_layer_idx
        if enable_segmentation:
            # 分割头：使用FPN的某一层特征进行分割
            # 使用最高分辨率层（索引0）以获得更好的空间细节
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(fpn_out_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),  # 输出单通道分割mask
                nn.Sigmoid()  # 类别无关，输出0-1的概率
            )
            
            # 初始化分割头权重
            self._initialize_segmentation_head()
        

    def _initialize_segmentation_head(self):
        """初始化分割头权重"""
        for m in self.segmentation_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        return_segmentation: bool = False
    ) -> dict:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            mask: mask张量 (B, 1, H, W)
            return_features: 是否返回backbone特征
            return_segmentation: 是否返回分割结果
            
        Returns:
            包含embedding和可选分割结果的字典
        """
        # Backbone特征提取（获取所有层）
        features = self.backbone(x, output_hidden_states=True)
        
        # FPN特征金字塔网络：优化多尺度特征图
        # FPN统一所有特征图的通道数，通过自顶向下和自底向上路径融合多尺度信息
        fpn_features = self.fpn(features)
        
        # 对比学习分支：特征融合（应用mask筛选）
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
        
        # 语义分割分支（与对比学习分支并行）
        if self.enable_segmentation:
            # 使用FPN的指定层进行分割（默认使用最高分辨率层）
            seg_feat = fpn_features[self.seg_layer_idx]  # (B, C, H, W)
            seg_logits = self.segmentation_head(seg_feat)  # (B, 1, H', W')
            
            # 如果分割输出尺寸与输入mask不一致，进行上采样
            if seg_logits.shape[2:] != mask.shape[2:]:
                seg_logits = torch.nn.functional.interpolate(
                    seg_logits,
                    size=mask.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            result['segmentation'] = seg_logits
        
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
