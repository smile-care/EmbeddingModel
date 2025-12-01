"""
SupCon模型：Backbone + Projection Head
"""
from typing import Optional

import torch
import torch.nn as nn

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
            pretrained=False if backbone_checkpoint else True,  # 不使用ImageNet预训练，使用SSL权重
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
        """
        加载backbone权重（从SSL训练的checkpoint中提取encoder部分）
        
        SSL checkpoint格式：
        - 'model_state_dict': 包含完整的MAE模型权重
          - 'encoder.xxx': backbone权重
          - 'decoder.xxx': decoder权重（不需要）
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            # SSL训练保存的标准格式
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 直接是state_dict
            state_dict = checkpoint
        
        # 从MAE模型中提取encoder（backbone）权重
        backbone_state_dict = {}
        for key, value in state_dict.items():
            # SSL训练的MAE模型中，encoder权重以'encoder.'开头
            if key.startswith('encoder.'):
                # 移除'encoder.'前缀，得到backbone的权重键
                new_key = key.replace('encoder.', '', 1)
                backbone_state_dict[new_key] = value
            # 如果权重键不包含decoder、head、projection等，可能是直接的backbone权重
            elif not any(x in key for x in ['decoder', 'head', 'projection', 'mask_token', 'decoder_embed', 'decoder_pos_embed', 'decoder_blocks', 'decoder_norm', 'decoder_pred']):
                # 可能是直接的backbone权重（兼容性处理）
                backbone_state_dict[key] = value
        
        # 加载权重
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_state_dict, strict=False
        )
        
        # 打印加载信息
        if missing_keys:
            print(f"警告：部分backbone权重未加载 ({len(missing_keys)}个): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"警告：部分backbone权重未使用 ({len(unexpected_keys)}个): {unexpected_keys[:5]}...")
        
        # 检查是否成功加载了权重
        if len(backbone_state_dict) == 0:
            print(f"错误：未能从checkpoint中提取backbone权重！")
            print(f"Checkpoint中的键: {list(state_dict.keys())[:10]}...")
        else:
            print(f"成功加载 {len(backbone_state_dict)} 个backbone权重参数")
    
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
        # Backbone特征提取
        # 统一使用forward()方法，与SSL训练时保持一致
        if hasattr(self.backbone, 'forward_features'):
            # timm模型（ViT, ConvNeXt）有forward_features方法
            features = self.backbone.forward_features(x)
        else:
            # ResNet等Sequential模型直接调用
            features = self.backbone(x)
        
        # 处理不同backbone的输出格式
        if isinstance(features, tuple):
            features = features[0]
        
        if features.dim() == 4:
            # CNN输出（ResNet, ConvNeXt）：全局平均池化
            features = nn.AdaptiveAvgPool2d(1)(features)
            features = features.view(features.size(0), -1)
        elif features.dim() == 3:
            # ViT输出：(B, N, D)，取CLS token或平均池化
            if features.size(1) > 1:
                # 有CLS token时，通常第一个是CLS token
                features = features[:, 0]  # 取CLS token
            else:
                features = features[:, 0]
        
        # Projection
        embeddings = self.projection_head(features)
        
        # 检查embeddings是否包含NaN或Inf
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            # 如果包含NaN/Inf，使用零填充并打印警告
            print("警告：Projection head输出包含NaN或Inf！")
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
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

