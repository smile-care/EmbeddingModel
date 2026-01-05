"""
监督对比学习数据集
基于data_config_zhenyu.yaml配置，支持mask信息
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy import ndimage
from torch.utils.data import ConcatDataset, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF


class MaskSoftDilation:
    """Mask软膨胀处理类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化mask软膨胀处理器
        
        Args:
            config: 配置字典，包含：
                - enabled: 是否启用（默认True）
        """
        if config is None:
            config = {}
        self.enabled = config.get('enabled', True)
    
    def __call__(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """
        对mask应用软膨胀
        
        Args:
            mask_tensor: mask张量 (1, H, W) 或 (C, H, W)，值在[0, 1]之间
            
        Returns:
            软膨胀后的mask张量，形状与输入相同
        """
        if not self.enabled:
            return mask_tensor
        
        # 确保mask是单通道的
        if mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[0:1]
        
        # 转换为numpy进行处理
        mask_np = mask_tensor.squeeze(0).cpu().numpy()  # (H, W)
        H, W = mask_np.shape
        
        # 二值化mask（前景为1，背景为0）
        binary_mask = (mask_np > 0.5).astype(np.float32)
        
        # 如果mask全为0，直接返回
        if binary_mask.sum() == 0:
            return mask_tensor
        
        # 根据mask前景区域的面积自适应计算膨胀半径
        # 计算前景区域的面积（像素数量）
        foreground_area = binary_mask.sum()
        
        if foreground_area == 0:
            # 如果没有前景区域，直接返回
            return mask_tensor
        
        # 假设前景区域是圆形的，根据面积计算等效直径
        # area = π * (diameter/2)^2 => diameter = 2 * sqrt(area / π)
        equivalent_diameter = np.sqrt(foreground_area / np.pi) * 2
        # 等效半径
        equivalent_radius = equivalent_diameter / 2
        
        # 直接使用等效直径作为膨胀半径
        # 这样无论前景区域是什么形状（细长、圆形、不规则），都能根据实际面积自适应
        dilation_radius = min(30, max(10, int(equivalent_radius)))
        
        # 计算距离变换：计算每个像素到最近前景像素的距离
        # 对于前景像素，距离为0；对于背景像素，距离为正数
        distance = ndimage.distance_transform_edt(1 - binary_mask)
        
        # 创建软膨胀mask，初始化为全0（背景）
        soft_mask = np.zeros_like(binary_mask, dtype=np.float32)
        
        # 找到膨胀区域：距离在(0, radius]范围内的背景像素
        # 注意：distance=0的像素（原始前景）不包含在内，确保原始前景保持不变
        dilation_region = (distance > 0) & (distance <= dilation_radius)
        
        # 在膨胀区域内，根据距离计算像素值
        # 距离为1时（紧邻前景），值接近1
        # 距离为radius时，值为0
        # 线性插值：value = 1 - (distance / radius)
        if dilation_region.any():
            normalized_distance = distance[dilation_region] / dilation_radius
            soft_mask[dilation_region] = 1.0 - normalized_distance
        
        # 确保原始前景区域保持为1
        soft_mask[binary_mask > 0.5] = 1.0
        
        # 转换回torch tensor
        soft_mask_tensor = torch.from_numpy(soft_mask).unsqueeze(0).to(mask_tensor.device)
        
        return soft_mask_tensor


class TwoViewAugmentation:
    """双视图数据增强（同时处理图像和mask）"""
    
    def __init__(self, config: Optional[Dict]):
        """
        初始化数据增强器
        
        Args:
            config: 增强配置字典
        """
        self.image_size = config.get('image_size', 224)
        
        # 构建增强pipeline（使用默认image_size，但会在__call__中动态更新）
        self.config = config
        self.transform1 = self._build_transform(config, view=1, image_size=self.image_size)
        self.transform2 = self._build_transform(config, view=2, image_size=self.image_size)
        
        # 图像归一化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # mask软膨胀处理器
        mask_dilation_config = config.get('mask_dilation', {})
        self.mask_dilation = MaskSoftDilation(mask_dilation_config)
    
    def _build_transform(self, config: Dict, view: int, image_size: int) -> transforms.Compose:
        """构建变换pipeline"""
        transform_list = []
        
        transform_list.append(
                transforms.Resize((image_size, image_size))
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
    
    def __call__(self, image: Image.Image, mask: Image.Image, image_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用双视图增强
        
        Args:
            image: PIL图像
            mask: PIL mask图像
            image_size: 动态指定的图像尺寸（如果提供，会覆盖初始化时的image_size）
            
        Returns:
            (view1_image, view1_mask, view2_image, view2_mask)
        """
        # 如果提供了动态image_size，重新构建transform
        current_image_size = image_size if image_size is not None else self.image_size
        if image_size is not None and image_size != self.image_size:
            transform1 = self._build_transform(self.config, view=1, image_size=current_image_size)
            transform2 = self._build_transform(self.config, view=2, image_size=current_image_size)
        else:
            transform1 = self.transform1
            transform2 = self.transform2
        
        # 对view1应用增强（确保image和mask使用相同的随机参数）
        view1_image, view1_mask = self._apply_augmentation_with_mask(image, mask, transform1)
        
        # 对view2应用增强（确保image和mask使用相同的随机参数）
        view2_image, view2_mask = self._apply_augmentation_with_mask(image, mask, transform2)
        
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
        
        # 应用软膨胀
        mask_tensor = self.mask_dilation(mask_tensor)
        
        return image_tensor, mask_tensor
    


class IndexWithScale:
    """包装索引和尺度的辅助类"""
    def __init__(self, idx: int, image_size: int):
        self.idx = idx
        self.image_size = image_size


class MultiScaleBatchSampler(Sampler):
    """
    多尺度Batch采样器
    确保每个batch内的所有样本使用相同的图像尺度
    """
    
    def __init__(self, dataset: Dataset, batch_size: int, image_sizes: List[int], shuffle: bool = True):
        """
        初始化多尺度Batch采样器
        
        Args:
            dataset: 数据集
            batch_size: batch大小
            image_sizes: 可用的图像尺度列表
            shuffle: 是否打乱数据顺序
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_sizes = image_sizes
        self.shuffle = shuffle
        
        # 创建索引列表
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        """生成batch索引（包装了尺度信息）"""
        if self.shuffle:
            # 打乱索引
            indices = self.indices.copy()
            random.shuffle(indices)
        else:
            indices = self.indices
        
        # 生成batch
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            # 为这个batch随机选择一个尺度
            image_size = random.choice(self.image_sizes)
            # 返回包装了尺度的索引对象列表
            yield [IndexWithScale(idx, image_size) for idx in batch_indices]
    
    def __len__(self):
        """返回batch数量"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def multi_scale_collate_fn(batch_data):
    """
    多尺度batch的collate函数
    
    Args:
        batch_data: 由MultiScaleBatchSampler和数据集返回的数据列表
                    每个元素是一个字典，包含view1_image, view1_mask等
    
    Returns:
        组织好的batch字典
    """
    # batch_data是一个列表，每个元素是一个样本字典
    # 将所有字段分别堆叠
    result = {}
    
    # 获取所有键
    keys = batch_data[0].keys()
    
    for key in keys:
        if key in ['view1_image', 'view1_mask', 'view2_image', 'view2_mask']:
            # 对于tensor字段，使用torch.stack
            result[key] = torch.stack([item[key] for item in batch_data])
        elif key in ['label']:
            # 对于label，使用torch.tensor
            result[key] = torch.tensor([item[key] for item in batch_data])
        else:
            # 对于其他字段（如字符串），保持列表格式
            result[key] = [item[key] for item in batch_data]
    
    return result


class SupConDataset(Dataset):
    """监督对比学习数据集（基于data_config_zhenyu.yaml）"""
    
    def __init__(
        self,
        data_config_path: str = 'configs/data_config_zhenyu.yaml',
        split: str = 'train',  # 'train' or 'val'
        image_size: Union[int, List[int]] = 224
    ):
        """
        初始化数据集
        
        Args:
            data_config_path: 数据配置文件路径
            split: 数据集划分（'train' 或 'val'）
            image_size: 输入图像大小，可以是单个整数或整数列表（多尺度训练）
        """
        # 加载配置
        with open(data_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.root = Path(self.config['root'])
        self.split = split
        
        # 处理image_size：统一转换为列表格式
        if isinstance(image_size, int):
            self.image_sizes = [image_size]
        elif isinstance(image_size, list):
            self.image_sizes = image_size
        else:
            raise ValueError(f"image_size必须是int或List[int]，当前为{type(image_size)}")
        
        # 为了向后兼容，保留image_size属性（使用第一个尺度）
        self.image_size = self.image_sizes[0]
        
        # 加载数据列表
        self.samples = self._load_samples()
        
        # 读取类别相似度配置
        self.default_similarity = self.config.get('default_similarity', 0.0)
        self.custom_similarity = self.config.get('custom_similarity', [])
        
        # 构建相似度矩阵
        self.similarity_matrix = self._build_similarity_matrix()
        
        # 构建增强器（使用第一个尺度作为默认值，实际使用时可以动态指定）
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
                'color_jitter': {'enabled': True},
                'mask_dilation': {
                    'enabled': True  # 是否启用mask软膨胀
                }
            }) if split == 'train' else None
        
        # mask软膨胀处理器（train和val都使用）
        mask_dilation_config = {
            'enabled': True
        }
        self.mask_dilation = MaskSoftDilation(mask_dilation_config)
        
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
        debug_mode = self.config.get('debug_mode', False)
        
        samples = []
        categories = set()
        
        # 遍历所有类别目录，使用rglob递归查找所有png文件
        for file_path in self.root.rglob("*.png"):
            # 仅处理当前划分的数据
            if data_split.get(self.split, self.split) != file_path.parent.parent.name and not debug_mode:
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
    
    def __getitem__(self, idx, current_image_size: Optional[int] = None):
        """
        获取数据样本
        
        Args:
            idx: 样本索引，可以是整数或IndexWithScale对象
            current_image_size: 当前使用的图像尺寸（用于多尺度训练，如果为None则使用默认尺度）
        """
        # 处理IndexWithScale包装的索引
        if isinstance(idx, IndexWithScale):
            actual_idx = idx.idx
            current_image_size = idx.image_size
        else:
            actual_idx = idx
        
        sample = self.samples[actual_idx]
        
        # 确定使用的图像尺寸
        if current_image_size is not None:
            image_size = current_image_size
        elif len(self.image_sizes) == 1:
            image_size = self.image_sizes[0]
        else:
            # 如果未指定且有多个尺度，随机选择一个（用于兼容性，但多尺度训练应通过BatchSampler控制）
            image_size = random.choice(self.image_sizes)
        
        # 加载图像和mask
        image = Image.open(sample['image_path']).convert('RGB')
        mask = Image.open(sample['mask_path']).convert('L')  # 灰度图
        
        # 应用增强
        if self.augmentation is not None:
            # 如果image_size与默认不同，需要动态更新
            view1_image, view1_mask, view2_image, view2_mask = self.augmentation(
                image, mask, image_size=image_size
            )
        else:
            # 验证集：只做resize和归一化
            # 如果image_size与默认不同，需要重新构建transform
            if image_size != self.image_size:
                val_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                val_mask_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ])
            else:
                val_transform = self.val_transform
                val_mask_transform = self.val_mask_transform
            
            view1_image = val_transform(image)
            view1_mask = val_mask_transform(mask)
            view1_mask = (view1_mask > 0.5).float()  # 二值化
            # 应用软膨胀（与train保持一致）
            view1_mask = self.mask_dilation(view1_mask)
            # 生成第二视图与第一视图相同
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
        image_size: Union[int, List[int]] = 224,
    ):
        """
        初始化多配置数据集
        
        Args:
            data_config_paths: 数据配置文件路径列表
            split: 数据集划分（'train' 或 'val'）
            image_size: 输入图像大小，可以是单个整数或整数列表（多尺度训练）
        """
        self.data_config_paths = data_config_paths
        self.split = split
        
        # 处理image_size：统一转换为列表格式
        if isinstance(image_size, int):
            self.image_sizes = [image_size]
        elif isinstance(image_size, list):
            self.image_sizes = image_size
        else:
            raise ValueError(f"image_size必须是int或List[int]，当前为{type(image_size)}")
        
        # 为了向后兼容，保留image_size属性（使用第一个尺度）
        self.image_size = self.image_sizes[0]
        
        # 为每个配置创建数据集
        self.datasets = []
        all_categories = []
        all_custom_similarity = []
        default_similarity = 0.0
        
        for config_path in data_config_paths:
            dataset = SupConDataset(
                data_config_path=config_path,
                split=split,
                image_size=image_size,  # 传递列表或整数
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
    
    def __getitem__(self, idx, current_image_size: Optional[int] = None):
        """
        获取数据样本
        
        Args:
            idx: 样本索引，可以是整数或IndexWithScale对象
            current_image_size: 当前使用的图像尺寸（用于多尺度训练，如果为None则使用默认尺度）
        """
        # 处理IndexWithScale包装的索引
        if isinstance(idx, IndexWithScale):
            actual_idx = idx.idx
            current_image_size = idx.image_size
        else:
            actual_idx = idx
        
        # 手动实现索引映射（因为需要重新映射类别索引）
        # 找到这个索引对应的数据集
        if actual_idx < 0:
            if -actual_idx > len(self):
                raise ValueError("索引超出范围")
            actual_idx = len(self) + actual_idx
        
        dataset_idx = 0
        cumulative_size = 0
        for i, dataset in enumerate(self.datasets):
            if actual_idx < cumulative_size + len(dataset):
                dataset_idx = i
                local_idx = actual_idx - cumulative_size
                break
            cumulative_size += len(dataset)
        else:
            raise IndexError(f"索引 {actual_idx} 超出范围")
        
        # 从对应的数据集获取样本（传递current_image_size）
        sample = self.datasets[dataset_idx].__getitem__(local_idx, current_image_size=current_image_size)
        
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
