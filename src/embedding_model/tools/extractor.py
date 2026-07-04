"""
Embedding提取工具
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..supcon.models.convnext_model import ConvNeXtModel
from ..utils.config_loader import load_config


class EmbeddingExtractor:
    """提取所有实例的embedding"""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化提取器
        
        Args:
            model_path: 模型checkpoint路径
            config_path: 配置文件路径（如果checkpoint中没有）
            device: 设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if config_path:
            config = load_config(config_path)
            model_config = config['supcon']['model']
            inference_config = config['supcon'].get('inference', {})
        else:
            model_config = checkpoint.get('config', {}).get('supcon', {}).get('model', {})
            inference_config = checkpoint.get('config', {}).get('supcon', {}).get('inference', {})
        self.embedding_source = model_config.get(
            'embedding_source',
            inference_config.get('embedding_source', 'representations')
            if isinstance(inference_config, dict)
            else 'representations'
        )
        
        # 创建模型
        self.model = ConvNeXtModel(
            backbone_cfg=None,  # 权重已在checkpoint中
            ckpt_path=None,
            embedding_dim=model_config.get('embedding_dim', 128),
            fusion_dim=model_config.get('convnext', {}).get('fusion_dim', model_config.get('embedding_dim', 128)),
            projection_hidden_dim=model_config.get('convnext', {}).get(
                'projection_hidden_dim',
                model_config.get('projection_hidden_dim')
            ),
            image_size=224,
            use_layers=model_config.get('convnext', {}).get('use_layers', [1, 2, 3]),
            mask_gating=model_config.get('convnext', {}).get('mask_gating', {}),
            pooling=model_config.get('convnext', {}).get('pooling', {'mode': 'fg_only'}),
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤掉classification_head的权重（如果模型没有分类头）
        # 因为提取embedding时不需要分类头
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('classification_head.'):
                filtered_state_dict[key] = value
        
        # 加载权重（使用strict=False以忽略不匹配的键）
        missing_keys, unexpected_keys = self.model.load_state_dict(
            filtered_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"警告：部分权重未加载: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"警告：部分权重未使用: {unexpected_keys[:5]}...")
        
        self.model.eval()
    
    def extract_from_image(self, image_path: str) -> np.ndarray:
        """
        从单张图像提取embedding
        
        Args:
            image_path: 图像路径
            
        Returns:
            Embedding向量
        """
        import torchvision.transforms as transforms
        from PIL import Image

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask = torch.ones(1, 1, 224, 224).to(self.device)
            outputs = self.model(image_tensor, mask, return_features=False)
            embedding = outputs[self.embedding_source].cpu().numpy()[0]
        
        return embedding
    
    def extract_batch(
        self,
        metadata_file: str,
        output_file: Optional[str] = None,
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        批量提取所有patch的embedding
        
        Args:
            metadata_file: patch元数据JSON文件路径
            output_file: 输出文件路径（.npy或.npz）
            batch_size: batch大小
            
        Returns:
            {instance_id: embedding} 字典
        """
        # 加载元数据
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        import torchvision.transforms as transforms
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset

        # 创建临时数据集
        class PatchDataset(Dataset):
            def __init__(self, patches, patch_root=None):
                self.patches = patches
                self.patch_root = Path(patch_root) if patch_root else None
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            
            def __len__(self):
                return len(self.patches)
            
            def __getitem__(self, idx):
                item = self.patches[idx]
                patch_path = item['patch_path']
                
                if self.patch_root and not Path(patch_path).is_absolute():
                    patch_path = self.patch_root / patch_path
                else:
                    patch_path = Path(patch_path)
                
                try:
                    image = Image.open(patch_path).convert('RGB')
                    image_tensor = self.transform(image)
                except:
                    print(f"加载图像失败 {patch_path}")
                    image_tensor = torch.zeros(3, 224, 224)
                
                return {
                    'image': image_tensor,
                    'instance_id': item['instance_id'],
                    'label': item['label'],
                    'domain_id': item['domain_id']
                }
        
        dataset = PatchDataset(metadata['patches'])
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 提取embedding
        embeddings_dict = {}
        all_embeddings = []
        all_instance_ids = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取embedding"):
                images = batch['image'].to(self.device)
                instance_ids = batch['instance_id']
                labels = batch['label']
                
                mask = torch.ones(images.shape[0], 1, images.shape[2], images.shape[3]).to(self.device)
                outputs = self.model(images, mask, return_features=False)
                embeddings = outputs[self.embedding_source].cpu().numpy()
                
                for i, instance_id in enumerate(instance_ids):
                    embeddings_dict[instance_id] = embeddings[i]
                    all_embeddings.append(embeddings[i])
                    all_instance_ids.append(instance_id)
                    all_labels.append(labels[i])
        
        # 保存
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix == '.npz':
                np.savez(
                    output_file,
                    embeddings=np.array(all_embeddings),
                    instance_ids=np.array(all_instance_ids),
                    labels=np.array(all_labels)
                )
            else:
                # 保存为字典格式
                np.save(output_file, embeddings_dict)
        
        return embeddings_dict


class MAEFeatureExtractor:
    """MAE特征提取器 - 提取MAE encoder的特征作为embedding"""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化MAE特征提取器
        
        Args:
            model_path: MAE模型checkpoint路径
            config_path: 配置文件路径（如果checkpoint中没有）
            device: 设备
        """
        from ..ssl_pretrain.models.mae import MAE
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if config_path:
            config = load_config(config_path)
            model_config = config['ssl']['model']
        else:
            model_config = checkpoint.get('config', {}).get('ssl', {}).get('model', {})
        
        # 创建MAE模型
        self.model = MAE(
            backbone_type=model_config.get('backbone', 'vit_base'),
            image_size=model_config.get('image_size', 224),
            patch_size=model_config.get('patch_size', 16),
            mask_ratio=model_config.get('mask_ratio', 0.75),
            decoder_dim=model_config.get('decoder_dim', 512),
            decoder_depth=model_config.get('decoder_depth', 8),
            decoder_num_heads=model_config.get('decoder_num_heads', 16),
            backbone_pretrained=False  # 使用checkpoint的权重
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        
        # 记录配置
        self.image_size = model_config.get('image_size', 224)
    
    def extract_from_image(self, image_path: str) -> np.ndarray:
        """
        从单张图像提取MAE encoder特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            Encoder特征向量
        """
        import torchvision.transforms as transforms
        from PIL import Image

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 使用MAE的encoder提取特征
            features = self.model.extract_features(image_tensor)  # (B, N, D)
            
            # 对所有patch特征进行平均池化，得到图像级别的特征
            embedding = features.mean(dim=1).cpu().numpy()[0]  # (D,)
        
        return embedding
    
    def extract_batch(
        self,
        metadata_file: str,
        output_file: Optional[str] = None,
        batch_size: int = 32,
        use_global_pooling: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        批量提取所有patch的MAE特征
        
        Args:
            metadata_file: patch元数据JSON文件路径
            output_file: 输出文件路径（.npy或.npz）
            batch_size: batch大小
            use_global_pooling: 是否对所有patch特征进行全局平均池化
                               True: 返回图像级特征 (D,)
                               False: 返回所有patch特征 (N, D)
            
        Returns:
            {instance_id: embedding} 字典
        """
        # 加载元数据
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        import torchvision.transforms as transforms
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset

        # 创建临时数据集
        class PatchDataset(Dataset):
            def __init__(self, patches, patch_root=None, image_size=224):
                self.patches = patches
                self.patch_root = Path(patch_root) if patch_root else None
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            
            def __len__(self):
                return len(self.patches)
            
            def __getitem__(self, idx):
                item = self.patches[idx]
                patch_path = item['patch_path']
                
                if self.patch_root and not Path(patch_path).is_absolute():
                    patch_path = self.patch_root / patch_path
                else:
                    patch_path = Path(patch_path)
                
                try:
                    image = Image.open(patch_path).convert('RGB')
                    image_tensor = self.transform(image)
                except Exception as e:
                    print(f"加载图像失败 {patch_path}: {e}")
                    image_tensor = torch.zeros(3, self.transform.transforms[0].size[0], 
                                               self.transform.transforms[0].size[1])
                
                return {
                    'image': image_tensor,
                    'instance_id': item['instance_id'],
                    'label': item['label'],
                    'domain_id': item['domain_id']
                }
        
        dataset = PatchDataset(
            metadata['patches'],
            image_size=self.image_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 提取特征
        embeddings_dict = {}
        all_embeddings = []
        all_instance_ids = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取MAE特征"):
                images = batch['image'].to(self.device)
                instance_ids = batch['instance_id']
                labels = batch['label']
                
                # 提取encoder特征
                features = self.model.extract_features(images)  # (B, N, D)
                
                if use_global_pooling:
                    # 全局平均池化，得到图像级特征
                    embeddings = features.mean(dim=1).cpu().numpy()  # (B, D)
                else:
                    # 保留所有patch特征
                    embeddings = features.cpu().numpy()  # (B, N, D)
                
                for i, instance_id in enumerate(instance_ids):
                    embeddings_dict[instance_id] = embeddings[i]
                    all_embeddings.append(embeddings[i])
                    all_instance_ids.append(instance_id)
                    all_labels.append(labels[i])
        
        # 保存
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix == '.npz':
                np.savez(
                    output_file,
                    embeddings=np.array(all_embeddings),
                    instance_ids=np.array(all_instance_ids),
                    labels=np.array(all_labels)
                )
                print(f"保存特征到: {output_file}")
                print(f"特征形状: {np.array(all_embeddings).shape}")
            else:
                # 保存为字典格式
                np.save(output_file, embeddings_dict)

        return embeddings_dict
