"""
SupCon模型：Backbone + Projection Head
"""
import torch
import torch.nn as nn
from typing import Optional
from ...ssl_pretrain.models.backbone_factory import BackboneFactory


class ProjectionHead(nn.Module):
    """Projection Head：将backbone特征映射到embedding空间"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (B, input_dim)
            
        Returns:
            Embedding (B, output_dim)
        """
        return self.projection(x)


class SupConModel(nn.Module):
    """SupCon模型"""
    
    def __init__(
        self,
        backbone_type: str,
        backbone_checkpoint: Optional[str] = None,
        embedding_dim: int = 128,
        projection_hidden_dims: list = [256, 128],
        num_classes: Optional[int] = None,
        image_size: int = 224,
        freeze_backbone: bool = False
    ):
        """
        初始化SupCon模型
        
        Args:
            backbone_type: backbone类型
            backbone_checkpoint: backbone权重路径（SSL阶段训练的模型）
            embedding_dim: embedding维度
            projection_hidden_dims: projection head隐藏层维度
            num_classes: 类别数（如果启用分类头）
            image_size: 输入图像大小
            freeze_backbone: 是否冻结backbone
        """
        super().__init__()
        
        # Backbone
        self.backbone = BackboneFactory.create_backbone(
            backbone_type,
            pretrained=False,  # 不使用ImageNet预训练，使用SSL权重
            image_size=image_size
        )
        
        # 加载SSL权重
        if backbone_checkpoint:
            self._load_backbone_weights(backbone_checkpoint)
        
        # 冻结backbone（如果需要）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 获取backbone输出维度
        backbone_dim = BackboneFactory.get_feature_dim(
            self.backbone, image_size
        )
        
        # Projection Head
        self.projection_head = ProjectionHead(
            input_dim=backbone_dim,
            hidden_dims=projection_hidden_dims,
            output_dim=embedding_dim,
            dropout=0.1
        )
        
        # 可选的分类头
        self.use_classification_head = num_classes is not None
        if self.use_classification_head:
            self.classification_head = nn.Linear(embedding_dim, num_classes)
        else:
            self.classification_head = None
    
    def _load_backbone_weights(self, checkpoint_path: str):
        """加载backbone权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 如果checkpoint包含encoder，提取encoder部分
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                backbone_state_dict[new_key] = value
            elif not any(x in key for x in ['decoder', 'head', 'projection']):
                backbone_state_dict[key] = value
        
        # 加载权重
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"警告：部分权重未加载: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"警告：部分权重未使用: {unexpected_keys[:5]}...")
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> dict:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            return_features: 是否返回backbone特征
            
        Returns:
            包含embedding和logits的字典
        """
        # Backbone特征
        features = self.backbone(x)
        
        # 处理不同backbone的输出格式
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 4:
            # CNN输出：全局平均池化
            features = nn.AdaptiveAvgPool2d(1)(features)
            features = features.view(features.size(0), -1)
        elif features.dim() == 3:
            # ViT输出：取CLS token或平均池化
            if features.size(1) > 1:
                features = features.mean(dim=1)
            else:
                features = features[:, 0]
        
        # Projection
        embeddings = self.projection_head(features)
        
        result = {'embeddings': embeddings}
        
        if return_features:
            result['features'] = features
        
        # 分类logits（如果启用）
        if self.use_classification_head:
            logits = self.classification_head(embeddings)
            result['logits'] = logits
        
        return result
    
    def freeze_backbone_layers(self, num_layers: Optional[int] = None):
        """
        冻结backbone的部分层
        
        Args:
            num_layers: 冻结的层数（None表示冻结全部）
        """
        if num_layers is None:
            # 冻结全部
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # 冻结前num_layers层
            layers = list(self.backbone.children())
            for i, layer in enumerate(layers[:num_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

