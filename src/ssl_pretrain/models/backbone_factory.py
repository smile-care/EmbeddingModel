"""
Backbone工厂：支持多种backbone架构
"""
import torch
import torch.nn as nn
import torchvision.models as models

try:
    import timm
except ImportError:
    timm = None


class BackboneFactory:
    """Backbone工厂类"""
    
    @staticmethod
    def create_backbone(
        backbone_type: str,
        pretrained: bool = True,
        image_size: int = 224,
        **kwargs
    ) -> nn.Module:
        """
        创建backbone模型
        
        Args:
            backbone_type: backbone类型 ('vit_base', 'resnet50', 'convnext_base')
            pretrained: 是否使用预训练权重
            image_size: 输入图像大小
            **kwargs: 其他参数
            
        Returns:
            Backbone模型
        """
        backbone_type = backbone_type.lower()
        
        if backbone_type.startswith('vit'):
            return BackboneFactory._create_vit(
                backbone_type, pretrained, image_size, **kwargs
            )
        elif backbone_type.startswith('resnet'):
            return BackboneFactory._create_resnet(
                backbone_type, pretrained, **kwargs
            )
        elif backbone_type.startswith('convnext'):
            return BackboneFactory._create_convnext(
                backbone_type, pretrained, **kwargs
            )
        else:
            raise ValueError(f"不支持的backbone类型: {backbone_type}")
    
    @staticmethod
    def _create_vit(
        backbone_type: str,
        pretrained: bool,
        image_size: int,
        **kwargs
    ) -> nn.Module:
        """创建Vision Transformer"""
        if timm is None:
            raise ImportError("timm is required for ViT backbone. Install it with: pip install timm")
        
        if backbone_type == 'vit_base':
            model_name = 'vit_base_patch16_224'
        elif backbone_type == 'vit_large':
            model_name = 'vit_large_patch16_224'
        elif backbone_type == 'vit_small':
            model_name = 'vit_small_patch16_224'
        else:
            model_name = 'vit_base_patch16_224'
        
        # 使用timm创建ViT
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=image_size,
            **kwargs
        )
        
        # 移除分类头，只保留特征提取部分
        if hasattr(model, 'head'):
            del model.head
        if hasattr(model, 'head_dist'):
            del model.head_dist
        
        return model
    
    @staticmethod
    def _create_resnet(
        backbone_type: str,
        pretrained: bool,
        **kwargs
    ) -> nn.Module:
        """创建ResNet"""
        if backbone_type == 'resnet50':
            model = models.resnet50(pretrained=pretrained, **kwargs)
        elif backbone_type == 'resnet101':
            model = models.resnet101(pretrained=pretrained, **kwargs)
        elif backbone_type == 'resnet34':
            model = models.resnet34(pretrained=pretrained, **kwargs)
        else:
            model = models.resnet50(pretrained=pretrained, **kwargs)
        
        # 移除分类头
        model = nn.Sequential(*list(model.children())[:-1])
        
        return model
    
    @staticmethod
    def _create_convnext(
        backbone_type: str,
        pretrained: bool,
        **kwargs
    ) -> nn.Module:
        """创建ConvNeXt"""
        if timm is None:
            raise ImportError("timm is required for ConvNeXt backbone. Install it with: pip install timm")
        
        if backbone_type == 'convnext_base':
            model_name = 'convnext_base'
        elif backbone_type == 'convnext_tiny':
            model_name = 'convnext_tiny'
        elif backbone_type == 'convnext_small':
            model_name = 'convnext_small'
        else:
            model_name = 'convnext_base'
        
        # 使用timm创建ConvNeXt
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            **kwargs
        )
        
        # 移除分类头
        if hasattr(model, 'head'):
            del model.head
        if hasattr(model, 'head_dist'):
            del model.head_dist
        
        return model
    
    @staticmethod
    def get_feature_dim(backbone: nn.Module, image_size: int = 224) -> int:
        """
        获取backbone的输出特征维度
        
        Args:
            backbone: backbone模型
            image_size: 输入图像大小
            
        Returns:
            特征维度
        """
        device = next(backbone.parameters()).device
        dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
        
        with torch.no_grad():
            # 对于timm模型，使用forward_features方法
            if hasattr(backbone, 'forward_features'):
                features = backbone.forward_features(dummy_input)
                # ViT输出: (B, N, D)，取CLS token
                if features.dim() == 3:
                    features = features[:, 0]  # 取CLS token
                # ConvNeXt输出: (B, D, H, W)
                elif features.dim() == 4:
                    features = nn.AdaptiveAvgPool2d(1)(features)
                    features = features.view(features.size(0), -1)
            else:
                # 对于其他模型（如ResNet Sequential）
                features = backbone(dummy_input)
                if isinstance(features, tuple):
                    features = features[0]
                if features.dim() == 4:
                    features = nn.AdaptiveAvgPool2d(1)(features)
                    features = features.view(features.size(0), -1)
                elif features.dim() == 3:
                    features = features.mean(dim=1) if features.size(1) > 1 else features[:, 0]
        
        return features.size(1)

