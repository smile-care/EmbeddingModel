"""
监督对比学习数据集
基于data_config_zhenyu.yaml配置，支持mask信息
"""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
import random


class TwoViewAugmentation:
    """双视图数据增强（同时处理图像和mask）"""
    
    def __init__(self, config: Optional[Dict]):
        """
        初始化数据增强器
        
        Args:
            config: 增强配置字典
        """
        self.image_size = config.get('image_size', 224)
        
        # 构建增强pipeline
        self.transform1 = self._build_transform(config, view=1)
        self.transform2 = self._build_transform(config, view=2)
        
        # 图像归一化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _build_transform(self, config: Dict, view: int) -> transforms.Compose:
        """构建变换pipeline"""
        transform_list = []
        
        transform_list.append(
                transforms.Resize((self.image_size, self.image_size))
            )
        
        # 水平翻转
        if config.get('horizontal_flip', {}).get('enabled', True):
            prob = config.get('horizontal_flip', {}).get('prob', 0.5)
            transform_list.append(transforms.RandomHorizontalFlip(p=prob))
        
        # 仿射变换（包含旋转、平移、剪切、缩放）
        if config.get('affine', {}).get('enabled', False):
            affine_config = config.get('affine', {})
            degrees = affine_config.get('degrees', 0)
            translate = affine_config.get('translate', (0.2, 0.2))  # (tx, ty) 平移比例
            scale = affine_config.get('scale', (0.8, 1.2))  # 缩放范围
            shear = affine_config.get('shear', 15)  # 剪切角度
            # fill: 填充值，默认0（黑色）。对于RGB图像可以是单个值或(R,G,B)元组
            fill = affine_config.get('fill', 0)
            transform_list.append(transforms.RandomAffine(
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                fill=fill
            ))
        
        # 颜色抖动
        if config.get('color_jitter', {}).get('enabled', False):
            transform_list.append(transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ))
        
        # 转为tensor
        transform_list.append(transforms.ToTensor())
        
        return transforms.Compose(transform_list)
    
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用双视图增强
        
        Args:
            image: PIL图像
            mask: PIL mask图像
            
        Returns:
            (view1_image, view1_mask, view2_image, view2_mask)
        """
        # 对view1应用增强（确保image和mask使用相同的随机参数）
        view1_image, view1_mask = self._apply_augmentation_with_mask(image, mask, self.transform1)
        
        # 对view2应用增强（确保image和mask使用相同的随机参数）
        view2_image, view2_mask = self._apply_augmentation_with_mask(image, mask, self.transform2)
        
        # 归一化图像
        view1_image = self.normalize(view1_image)
        view2_image = self.normalize(view2_image)
        
        return view1_image, view1_mask, view2_image, view2_mask
    
    def _apply_augmentation_with_mask(self, image: Image.Image, mask: Image.Image, transform: transforms.Compose) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对image和mask应用相同的空间变换，确保随机参数一致
        
        Returns:
            (transformed_image_tensor, transformed_mask_tensor)
        """
        # 保存随机状态，确保image和mask使用相同的随机参数
        random_state = random.getstate()
        torch_state = torch.get_rng_state()
        
        # 应用变换到图像（包含所有变换，包括颜色变换）
        transformed_image = image
        for t in transform.transforms:
            if isinstance(t, transforms.ToTensor):
                break
            transformed_image = t(transformed_image)
        
        # 恢复随机状态，确保mask使用相同的随机参数
        random.setstate(random_state)
        torch.set_rng_state(torch_state)
        
        # 对mask应用相同的空间变换（跳过颜色变换）
        transformed_mask = mask
        for t in transform.transforms:
            if isinstance(t, transforms.ToTensor):
                break
            # 跳过颜色变换
            if isinstance(t, transforms.ColorJitter):
                continue
            transformed_mask = t(transformed_mask)
        
        # 转为tensor
        image_tensor = transforms.ToTensor()(transformed_image)
        mask_tensor = transforms.ToTensor()(transformed_mask)
        
        # 确保mask与图像尺寸一致
        if mask_tensor.shape[1:] != image_tensor.shape[1:]:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=image_tensor.shape[1:],
                mode='nearest'
            ).squeeze(0)
        
        # 确保mask是二值的
        if mask_tensor.shape[0] == 1:
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = (mask_tensor[0:1] > 0.5).float()
        
        return image_tensor, mask_tensor
    


class SupConDataset(Dataset):
    """监督对比学习数据集（基于data_config_zhenyu.yaml）"""
    
    def __init__(
        self,
        data_config_path: str = 'configs/data_config_zhenyu.yaml',
        split: str = 'train',  # 'train' or 'val'
        image_size: int = 224
    ):
        """
        初始化数据集
        
        Args:
            data_config_path: 数据配置文件路径
            split: 数据集划分（'train' 或 'val'）
            image_size: 输入图像大小
        """
        # 加载配置
        with open(data_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.root = Path(self.config['root'])
        self.split = split
        self.image_size = image_size
        
        # 加载数据列表
        self.samples = self._load_samples()
        
        # 读取类别相似度配置
        self.default_similarity = self.config.get('default_similarity', 0.0)
        self.custom_similarity = self.config.get('custom_similarity', [])
        
        # 构建相似度矩阵
        self.similarity_matrix = self._build_similarity_matrix()
        
        # 构建增强器
        self.augmentation = TwoViewAugmentation({
                'image_size': self.image_size,
                'horizontal_flip': {'enabled': True, 'prob': 0.5},
                'affine': {
                    'enabled': True,
                    'degrees': 15,  # 旋转角度范围
                    'translate': (0.2, 0.2),  # 平移比例 (tx, ty)
                    'scale': (0.8, 1.2),  # 缩放范围
                    'shear': 20,  # 剪切角度
                    'fill': 0  # 填充值：0=黑色填充（默认）
                },
                'color_jitter': {'enabled': True}
            }) if split == 'train' else None
        
        # 验证集不使用增强，只做resize和归一化
        if split == 'val' and self.augmentation is None:
            self.val_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            self.val_mask_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()
            ])
    
    def _build_similarity_matrix(self) -> np.ndarray:
        """构建类别相似度矩阵"""
        num_categories = len(self.categories)
        similarity_matrix = np.full((num_categories, num_categories), self.default_similarity, dtype=float)
        
        # 设置对角线为1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # 应用自定义相似度
        for item in self.custom_similarity:
            categories_list = item['list']
            sim = item['similarity']
            # 为列表中的所有类别对设置相似度
            for i, cat1 in enumerate(categories_list):
                for cat2 in categories_list[i+1:]:
                    if cat1 not in self.cat2idx or cat2 not in self.cat2idx:
                        print(f"警告: 跳过未知类别 {cat1} 或 {cat2}")
                        continue
                    idx1 = self.cat2idx[cat1]
                    idx2 = self.cat2idx[cat2]
                    similarity_matrix[idx1, idx2] = sim
                    similarity_matrix[idx2, idx1] = sim  # 对称矩阵
        
        return similarity_matrix
    
    def _load_samples(self) -> list:
        """加载数据样本列表"""
        if not self.root.exists():
            raise FileNotFoundError(f"数据根目录不存在: {self.root}")
        
        data_split = self.config.get('data_split', {'train': 'train', 'val': 'val'})
        
        samples = []
        categories = set()
        
        # 遍历所有类别目录，使用rglob递归查找所有png文件
        for file_path in self.root.rglob("*.png"):
            # 仅处理当前划分的数据
            if data_split.get(self.split, self.split) != file_path.parent.parent.name:
                continue
            # 跳过mask文件
            if "_mask.png" in file_path.name:
                continue
            
            # 检查对应的mask文件是否存在
            mask_path = file_path.with_name(file_path.stem + "_mask.png")
            if not mask_path.exists():
                print(f"警告: 缺失掩码文件: {mask_path}，跳过该样本")
                continue
            
            # 获取类别（父目录名）
            category = file_path.parent.name
            categories.add(category)
            
            samples.append({
                'image_path': str(file_path),
                'mask_path': str(mask_path),
                'label': category,
                'label_idx': None,  # 占位符，稍后映射
            })
        
        # 构建类别映射
        self.categories = sorted(list(categories))
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}
        # 映射类别索引
        for sample in samples:
            sample['label_idx'] = self.cat2idx[sample['label']]
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像和mask
        image = Image.open(sample['image_path']).convert('RGB')
        mask = Image.open(sample['mask_path']).convert('L')  # 灰度图
        
        # 应用增强
        if self.augmentation is not None:
            view1_image, view1_mask, view2_image, view2_mask = self.augmentation(image, mask)
        else:
            # 验证集：只做resize和归一化
            view1_image = self.val_transform(image)
            view1_mask = self.val_mask_transform(mask)
            view1_mask = (view1_mask > 0.5).float()  # 二值化
            view2_image = view1_image.clone()
            view2_mask = view1_mask.clone()
        
        return {
            'view1_image': view1_image,
            'view1_mask': view1_mask,
            'view2_image': view2_image,
            'view2_mask': view2_mask,
            'label': sample['label_idx'],
            'label_name': sample['label'],
            'image_path': sample['image_path'],
            'mask_path': sample['mask_path'],
        }
    
    def get_similarity_matrix(self) -> np.ndarray:
        """获取相似度矩阵"""
        return self.similarity_matrix.copy()


class MultiConfigDataset(Dataset):
    """合并多个数据配置的数据集包装类"""
    
    def __init__(
        self,
        data_config_paths: list,
        split: str = 'train',
        image_size: int = 224,
    ):
        """
        初始化多配置数据集
        
        Args:
            data_config_paths: 数据配置文件路径列表
            split: 数据集划分（'train' 或 'val'）
            image_size: 输入图像大小
        """
        self.data_config_paths = data_config_paths
        self.split = split
        
        # 为每个配置创建数据集
        self.datasets = []
        all_categories = []
        all_custom_similarity = []
        default_similarity = 0.0
        
        for config_path in data_config_paths:
            dataset = SupConDataset(
                data_config_path=config_path,
                split=split,
                image_size=image_size,
            )
            self.datasets.append(dataset)
            
            # 收集所有类别（去重）
            for cat in dataset.categories:
                if cat not in all_categories:
                    all_categories.append(cat)
            
            # 收集自定义相似度
            all_custom_similarity.extend(dataset.custom_similarity)
            
            # 使用第一个数据集的默认相似度
            if len(self.datasets) == 1:
                default_similarity = dataset.default_similarity
        
        # 合并后的类别列表
        self.categories = all_categories
        self.default_similarity = default_similarity
        self.custom_similarity = all_custom_similarity
        
        # 构建合并后的类别映射
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}
        
        # 为每个数据集重新映射类别索引
        self.dataset_label_mappings = []
        for dataset in self.datasets:
            label_mapping = {}
            for old_idx, cat in dataset.idx2cat.items():
                new_idx = self.cat2idx[cat]
                label_mapping[old_idx] = new_idx
            self.dataset_label_mappings.append(label_mapping)
        
        # 构建合并后的相似度矩阵
        self.similarity_matrix = self._build_merged_similarity_matrix()
        
        # 使用 ConcatDataset 合并数据集
        self.concat_dataset = ConcatDataset(self.datasets)
    
    def _build_merged_similarity_matrix(self) -> np.ndarray:
        """构建合并后的相似度矩阵"""
        num_categories = len(self.categories)
        similarity_matrix = np.full((num_categories, num_categories), self.default_similarity, dtype=float)
        
        # 设置对角线为1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # 应用自定义相似度
        for item in self.custom_similarity:
            categories_list = item['list']
            sim = item['similarity']
            # 为列表中的所有类别对设置相似度
            for i, cat1 in enumerate(categories_list):
                for cat2 in categories_list[i+1:]:
                    if cat1 not in self.cat2idx or cat2 not in self.cat2idx:
                        continue
                    idx1 = self.cat2idx[cat1]
                    idx2 = self.cat2idx[cat2]
                    similarity_matrix[idx1, idx2] = sim
                    similarity_matrix[idx2, idx1] = sim  # 对称矩阵
        
        return similarity_matrix
    
    def __len__(self):
        return len(self.concat_dataset)
    
    def __getitem__(self, idx):
        # ConcatDataset 会自动处理索引映射
        # 我们需要找到这个索引对应的数据集
        if idx < 0:
            if -idx > len(self):
                raise ValueError("索引超出范围")
            idx = len(self) + idx
        
        dataset_idx = 0
        cumulative_size = 0
        for i, dataset in enumerate(self.datasets):
            if idx < cumulative_size + len(dataset):
                dataset_idx = i
                local_idx = idx - cumulative_size
                break
            cumulative_size += len(dataset)
        else:
            raise IndexError(f"索引 {idx} 超出范围")
        
        # 从对应的数据集获取样本
        sample = self.datasets[dataset_idx][local_idx]
        
        # 重新映射类别索引
        old_label_idx = sample['label']
        new_label_idx = self.dataset_label_mappings[dataset_idx][old_label_idx]
        
        # 创建新的样本，使用新的类别索引
        new_sample = sample.copy()
        new_sample['label'] = new_label_idx
        
        return new_sample
    
    def get_similarity_matrix(self) -> np.ndarray:
        """获取相似度矩阵"""
        return self.similarity_matrix.copy()
