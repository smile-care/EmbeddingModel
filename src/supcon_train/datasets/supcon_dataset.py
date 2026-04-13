"""
监督对比学习数据集
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


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
            color_jitter_config = config.get('color_jitter', {})
            brightness = color_jitter_config.get('brightness', 0.1)
            contrast = color_jitter_config.get('contrast', 0.1)
            saturation = color_jitter_config.get('saturation', 0.1)
            hue = color_jitter_config.get('hue', 0.05)
            transform_list.append(transforms.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
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
    
    def __init__(self, dataset: Dataset, batch_size: int, image_sizes: List[int], shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_sizes = image_sizes
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        indices = self.indices.copy()
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            image_size = random.choice(self.image_sizes)
            yield [IndexWithScale(idx, image_size) for idx in batch_indices]

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
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
    """监督对比学习数据集"""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: Union[int, List[int]] = 224,
        name: str = '',
        mask_dilation_config: Optional[Dict] = None,
    ):
        """
        Args:
            root: 数据目录，结构为 root/<category>/<image>.png
            split: 'train' 使用数据增强，'val' 仅做 resize + 归一化
            image_size: 输入尺寸，单个整数或整数列表（多尺度训练）
            name: 数据集名称，仅用于日志
            mask_dilation_config: mask软膨胀配置（来自config文件）
        """
        self.root = Path(root)
        self.split = split
        self.name = name

        if isinstance(image_size, int):
            self.image_sizes = [image_size]
        elif isinstance(image_size, list):
            self.image_sizes = image_size
        else:
            raise ValueError(f"image_size 必须是 int 或 List[int]，当前为 {type(image_size)}")
        self.image_size = self.image_sizes[0]

        self.samples = self._load_samples()

        # 获取mask_dilation配置（如果未提供则使用默认值）
        if mask_dilation_config is None:
            mask_dilation_config = {'enabled': False}

        self.augmentation = TwoViewAugmentation({
            'image_size': self.image_size,
            'horizontal_flip': {'enabled': True, 'prob': 0.5},
            'affine': {
                'enabled': True,
                'degrees': 15,
                'translate': (0.2, 0.2),
                'scale': (0.8, 1.2),
                'shear': 15,
                'fill': 0,
            },
            'color_jitter': {
                'enabled': True,
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.1,
                'hue': 0.05,
            },
            'mask_dilation': mask_dilation_config,
        }) if split == 'train' else None

        self.mask_dilation = MaskSoftDilation(mask_dilation_config)

        if split == 'val':
            self.val_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.val_mask_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
    
    def _load_samples(self) -> list:
        """扫描 root 下所有图像，构建样本列表与类别映射"""
        if not self.root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.root}")

        samples = []
        categories = set()

        for file_path in self.root.rglob("*.png"):
            if "_mask.png" in file_path.name:
                continue
            mask_path = file_path.with_name(file_path.stem + "_mask.png")
            if not mask_path.exists():
                print(f"警告: 缺失掩码文件: {mask_path}，跳过该样本")
                continue
            category = file_path.parent.name
            categories.add(category)
            samples.append({
                'image_path': str(file_path),
                'mask_path': str(mask_path),
                'label': category,
                'label_idx': None,
            })

        self.categories = sorted(list(categories))
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}
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

