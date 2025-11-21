"""
监督对比学习数据集
"""
import json
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
from ...ssl_pretrain.augmentations.industrial_aug import TwoViewAugmentation


class SupConDataset(Dataset):
    """监督对比学习数据集"""
    
    def __init__(
        self,
        metadata_file: str,
        patch_root: Optional[str] = None,
        augmentation_config: Optional[dict] = None
    ):
        """
        初始化数据集
        
        Args:
            metadata_file: patch元数据JSON文件路径
            patch_root: patch根目录（如果元数据中是相对路径）
            augmentation_config: 数据增强配置
        """
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.patch_root = Path(patch_root) if patch_root else None
        self.patches = metadata['patches']
        
        # 构建标签映射
        self.labels = sorted(list(set(p['label'] for p in self.patches)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 构建增强器
        if augmentation_config:
            self.augmentation = TwoViewAugmentation(augmentation_config)
        else:
            # 默认增强
            self.augmentation = TwoViewAugmentation({
                'image_size': 224,
                'random_crop': {'enabled': True, 'scale': [0.6, 1.0]},
                'horizontal_flip': {'enabled': True, 'prob': 0.5},
                'color_jitter': {'enabled': True}
            })
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        item = self.patches[idx]
        patch_path = item['patch_path']
        
        # 构建完整路径
        if self.patch_root and not Path(patch_path).is_absolute():
            patch_path = self.patch_root / patch_path
        else:
            patch_path = Path(patch_path)
        
        # 加载图像
        try:
            image = Image.open(patch_path).convert('RGB')
        except Exception as e:
            # 如果加载失败，返回黑色图像
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # 应用双视图增强
        view1, view2 = self.augmentation(image)
        
        # 标签
        label = item['label']
        label_idx = self.label_to_idx[label]
        
        return {
            'view1': view1,
            'view2': view2,
            'label': label_idx,
            'label_name': label,
            'domain_id': item['domain_id'],
            'instance_id': item['instance_id']
        }
    
    def get_label_balanced_sampler(
        self,
        batch_size: int,
        samples_per_class: int = 4
    ):
        """
        获取label平衡采样器（保证每个batch内每个label有多个样本）
        
        Args:
            batch_size: batch大小
            samples_per_class: 每个类别在每个batch中的样本数
            
        Returns:
            自定义采样器（需要配合BatchSampler使用）
        """
        # 按label组织样本索引
        label_indices = defaultdict(list)
        for idx, item in enumerate(self.patches):
            label = item['label']
            label_indices[label].append(idx)
        
        # 构建采样索引列表
        all_indices = []
        labels_list = list(label_indices.keys())
        
        # 计算需要多少个batch
        n_batches = len(self.patches) // batch_size + 1
        
        for _ in range(n_batches):
            batch_indices = []
            
            # 为每个label采样samples_per_class个样本
            for label in labels_list:
                indices = label_indices[label]
                if len(indices) >= samples_per_class:
                    selected = np.random.choice(
                        indices, samples_per_class, replace=False
                    ).tolist()
                else:
                    # 如果样本数不足，使用有放回采样
                    selected = np.random.choice(
                        indices, samples_per_class, replace=True
                    ).tolist()
                batch_indices.extend(selected)
            
            # 如果batch还没满，随机补充
            if len(batch_indices) < batch_size:
                remaining = batch_size - len(batch_indices)
                all_remaining_indices = [
                    idx for idx in range(len(self.patches))
                    if idx not in batch_indices
                ]
                if len(all_remaining_indices) >= remaining:
                    additional = np.random.choice(
                        all_remaining_indices, remaining, replace=False
                    ).tolist()
                else:
                    additional = np.random.choice(
                        range(len(self.patches)), remaining, replace=True
                    ).tolist()
                batch_indices.extend(additional)
            
            # 打乱
            np.random.shuffle(batch_indices)
            all_indices.extend(batch_indices[:batch_size])
        
        return all_indices

