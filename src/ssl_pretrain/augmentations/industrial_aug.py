"""
工业图像增强策略
"""
from typing import Dict

import torch
import torchvision.transforms as transforms
from PIL import Image


class IndustrialAugmentation:
    """工业图像专用数据增强"""
    
    def __init__(self, config: Dict):
        """
        初始化增强器
        
        Args:
            config: 增强配置字典
        """
        self.config = config
        self.transform = self._build_transform()
    
    def _build_transform(self) -> transforms.Compose:
        """构建增强pipeline"""
        transform_list = []
        
        # 随机裁剪
        if self.config.get('random_crop', {}).get('enabled', False):
            scale = self.config['random_crop'].get('scale', [0.4, 1.0])
            transform_list.append(
                transforms.RandomResizedCrop(
                    size=self.config.get('image_size', 224),
                    scale=tuple(scale),
                    ratio=(0.75, 1.33)
                )
            )
        else:
            transform_list.append(
                transforms.Resize((self.config.get('image_size', 224),) * 2)
            )
        
        # 水平翻转
        if self.config.get('horizontal_flip', {}).get('enabled', False):
            prob = self.config['horizontal_flip'].get('prob', 0.5)
            transform_list.append(transforms.RandomHorizontalFlip(p=prob))
        
        # 垂直翻转
        if self.config.get('vertical_flip', {}).get('enabled', False):
            prob = self.config['vertical_flip'].get('prob', 0.3)
            transform_list.append(transforms.RandomVerticalFlip(p=prob))
        
        # 旋转
        if self.config.get('rotation', {}).get('enabled', False):
            degrees = self.config['rotation'].get('degrees', 15)
            transform_list.append(transforms.RandomRotation(degrees=degrees))
        
        # 颜色抖动（轻度）
        if self.config.get('color_jitter', {}).get('enabled', False):
            brightness = self.config['color_jitter'].get('brightness', 0.1)
            contrast = self.config['color_jitter'].get('contrast', 0.1)
            saturation = self.config['color_jitter'].get('saturation', 0.05)
            hue = self.config['color_jitter'].get('hue', 0.02)
            transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                )
            )
        
        # 转换为Tensor
        transform_list.append(transforms.ToTensor())
        
        # 归一化（ImageNet标准）
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
        
        # 高斯噪声（在Tensor上添加）
        if self.config.get('gaussian_noise', {}).get('enabled', False):
            # 注意：需要在ToTensor之后添加，所以单独处理
            pass
        
        return transforms.Compose(transform_list)
    
    def add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        添加高斯噪声
        
        Args:
            tensor: 输入tensor (C, H, W)
            
        Returns:
            添加噪声后的tensor
        """
        if not self.config.get('gaussian_noise', {}).get('enabled', False):
            return tensor
        
        std = self.config['gaussian_noise'].get('std', 0.01)
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0, 1)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        应用增强
        
        Args:
            image: PIL图像
            
        Returns:
            增强后的tensor
        """
        tensor = self.transform(image)
        
        # 添加高斯噪声（如果需要）
        if self.config.get('gaussian_noise', {}).get('enabled', False):
            tensor = self.add_gaussian_noise(tensor)
        
        return tensor


class TwoViewAugmentation:
    """双视图增强（用于对比学习）"""
    
    def __init__(self, config: Dict):
        """
        初始化双视图增强器
        
        Args:
            config: 增强配置
        """
        self.config = config
        self.base_transform = IndustrialAugmentation(config)
    
    def __call__(self, image: Image.Image) -> tuple:
        """
        生成两个增强视图
        
        Args:
            image: PIL图像
            
        Returns:
            (view1, view2) 两个增强后的tensor
        """
        view1 = self.base_transform(image)
        view2 = self.base_transform(image)
        return view1, view2

