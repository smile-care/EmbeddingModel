"""
自监督学习数据集
"""
import json
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import defaultdict
import numpy as np
from ..augmentations.industrial_aug import IndustrialAugmentation


class SSLDataset(Dataset):
    """自监督学习数据集"""
    
    def __init__(
        self,
        metadata_file: Union[str, List[str]],
        image_root: Optional[str] = None,
        augmentation_config: Optional[dict] = None,
        domain_balanced: bool = True
    ):
        """
        初始化数据集
        
        Args:
            metadata_file: 元数据JSON文件路径，可以是单个文件路径或文件路径列表（用于合并多个数据集）
            image_root: 图像根目录（如果元数据中是相对路径，已废弃，因为现在使用绝对路径）
            augmentation_config: 数据增强配置
            domain_balanced: 是否使用domain平衡采样
        """
        # 支持单个文件或文件列表
        if isinstance(metadata_file, str):
            metadata_files = [metadata_file]
        else:
            metadata_files = metadata_file
        
        # 合并多个数据集
        all_images = []
        all_domains = set()
        total_images = 0
        
        for meta_file in metadata_files:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            all_images.extend(metadata['images'])
            all_domains.update(metadata.get('domains', []))
            total_images += metadata.get('total_images', len(metadata['images']))
        
        self.image_root = Path(image_root) if image_root else None
        self.images = all_images
        self.domain_balanced = domain_balanced
        
        # 构建增强器
        if augmentation_config:
            self.augmentation = IndustrialAugmentation(augmentation_config)
        else:
            # 默认增强
            self.augmentation = IndustrialAugmentation({
                'image_size': 224,
                'random_crop': {'enabled': True, 'scale': [0.4, 1.0]},
                'horizontal_flip': {'enabled': True, 'prob': 0.5},
                'color_jitter': {'enabled': True}
            })
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        image_path = item['image_path']
        
        # 构建完整路径
        image_path = Path(image_path)
        
        # 加载图像
        assert image_path.exists(), f"图像文件不存在: {image_path}"
        image = Image.open(image_path).convert('RGB')
        assert image is not None, f"无法打开图像文件: {image_path}"
        
        # 应用增强
        tensor = self.augmentation(image)
        
        return {
            'image': tensor,
            'domain_id': item['domain_id'],
            'image_path': str(image_path)
        }
    
    def get_domain_balanced_sampler(self):
        """
        获取domain平衡采样器
        
        Returns:
            WeightedRandomSampler
        """
        if not self.domain_balanced:
            return None
        
        # 统计每个domain的样本数
        domain_counts = defaultdict(int)
        for item in self.images:
            domain_counts[item['domain_id']] += 1
        
        # 计算权重（样本数少的domain权重高）
        total_samples = len(self.images)
        n_domains = len(domain_counts)
        weights = []
        
        for item in self.images:
            domain_id = item['domain_id']
            weight = total_samples / (n_domains * domain_counts[domain_id])
            weights.append(weight)
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

