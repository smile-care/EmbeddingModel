"""
Embedding提取工具
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import json

from ..supcon_train.models.supcon_model import SupConModel
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
        else:
            model_config = checkpoint.get('config', {}).get('supcon', {}).get('model', {})
        
        # 创建模型
        self.model = SupConModel(
            backbone_type=model_config.get('backbone', 'vit_base'),
            backbone_checkpoint=None,  # 权重已在checkpoint中
            embedding_dim=model_config.get('embedding_dim', 128),
            projection_hidden_dims=model_config.get('projection_head', {}).get('hidden_dims', [256, 128]),
            num_classes=None,  # 不需要分类头
            image_size=224
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def extract_from_image(self, image_path: str) -> np.ndarray:
        """
        从单张图像提取embedding
        
        Args:
            image_path: 图像路径
            
        Returns:
            Embedding向量
        """
        from PIL import Image
        import torchvision.transforms as transforms
        
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
            outputs = self.model(image_tensor, return_features=False)
            embedding = outputs['embeddings'].cpu().numpy()[0]
        
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
        
        from PIL import Image
        import torchvision.transforms as transforms
        from torch.utils.data import Dataset, DataLoader
        
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
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取embedding"):
                images = batch['image'].to(self.device)
                instance_ids = batch['instance_id']
                
                outputs = self.model(images, return_features=False)
                embeddings = outputs['embeddings'].cpu().numpy()
                
                for i, instance_id in enumerate(instance_ids):
                    embeddings_dict[instance_id] = embeddings[i]
                    all_embeddings.append(embeddings[i])
                    all_instance_ids.append(instance_id)
        
        # 保存
        if output_file:
            output_path = Path(output_file)
            if output_path.suffix == '.npz':
                np.savez(
                    output_file,
                    embeddings=np.array(all_embeddings),
                    instance_ids=np.array(all_instance_ids)
                )
            else:
                # 保存为字典格式
                np.save(output_file, embeddings_dict)
        
        return embeddings_dict

