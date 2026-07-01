"""
模型组件模块
包含特征融合、mask pooling、投影头等组件
"""
from .feature_fusion import FeatureFusion
from .mask_pooling import MaskWeightedPooling
from .projection_head import ProjectionHead

__all__ = [
    'FeatureFusion',
    'MaskWeightedPooling',
    'ProjectionHead',
]
