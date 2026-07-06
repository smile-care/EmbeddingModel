"""监督对比学习数据集。"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .augmentations import MaskSoftDilation, TwoViewAugmentation
from .samplers import IndexWithScale, MultiScaleBatchSampler, multi_scale_collate_fn

__all__ = [
    'SupConDataset',
    'MaskSoftDilation',
    'TwoViewAugmentation',
    'IndexWithScale',
    'MultiScaleBatchSampler',
    'multi_scale_collate_fn',
]


class SupConDataset(Dataset):
    """监督对比学习数据集。"""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: Union[int, List[int]] = 224,
        name: str = '',
        mask_dilation_config: Optional[Dict] = None,
        copy_paste_config: Optional[Dict] = None,
    ):
        """
        Args:
            root: 数据目录，结构为 root/<category>/<image>.png
            split: 'train' 使用数据增强，'val' 仅做 resize + 归一化
            image_size: 输入尺寸，单个整数或整数列表（多尺度训练）
            name: 数据集名称，仅用于日志
            mask_dilation_config: mask软膨胀配置
            copy_paste_config: mask 外缺陷干扰 copy-paste 增强配置
        """
        self.root = Path(root)
        self.split = split
        self.name = name
        self.image_sizes = self._normalize_image_sizes(image_size)
        self.image_size = self.image_sizes[0]
        self.samples = self._load_samples()

        mask_dilation_config = mask_dilation_config or {'enabled': False}
        copy_paste_config = copy_paste_config or {'enabled': False}
        self.mask_dilation = MaskSoftDilation(mask_dilation_config)
        self.augmentation = self._build_train_augmentation(
            mask_dilation_config,
            copy_paste_config,
        ) if split == 'train' else None

        if split == 'val':
            self.val_transform = self._build_val_image_transform(self.image_size)
            self.val_mask_transform = self._build_val_mask_transform(self.image_size)

    @staticmethod
    def _normalize_image_sizes(image_size: Union[int, List[int]]) -> List[int]:
        if isinstance(image_size, int):
            return [image_size]
        if isinstance(image_size, list):
            return image_size
        raise ValueError(f"image_size 必须是 int 或 List[int]，当前为 {type(image_size)}")

    def _build_train_augmentation(
        self,
        mask_dilation_config: Dict,
        copy_paste_config: Dict,
    ) -> TwoViewAugmentation:
        return TwoViewAugmentation({
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
            'copy_paste': copy_paste_config,
            'copy_paste_samples': self.samples,
        })

    @staticmethod
    def _build_val_image_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _build_val_mask_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _load_samples(self) -> list:
        """扫描 root 下所有图像，构建样本列表与类别映射。"""
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

        self.categories = sorted(categories)
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}
        for sample in samples:
            sample['label_idx'] = self.cat2idx[sample['label']]
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, current_image_size: Optional[int] = None):
        actual_idx, image_size = self._resolve_index_and_size(idx, current_image_size)
        sample = self.samples[actual_idx]

        image = Image.open(sample['image_path']).convert('RGB')
        mask = Image.open(sample['mask_path']).convert('L')

        if self.augmentation is not None:
            view1_image, view1_mask, view2_image, view2_mask = self.augmentation(
                image,
                mask,
                image_size=image_size,
                current_sample=sample,
            )
        else:
            view1_image, view1_mask = self._build_val_view(image, mask, image_size)
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

    def _resolve_index_and_size(self, idx, current_image_size: Optional[int]) -> tuple[int, int]:
        if isinstance(idx, IndexWithScale):
            actual_idx = idx.idx
            current_image_size = idx.image_size
        else:
            actual_idx = idx

        if current_image_size is not None:
            image_size = current_image_size
        elif len(self.image_sizes) == 1:
            image_size = self.image_sizes[0]
        else:
            image_size = random.choice(self.image_sizes)
        return actual_idx, image_size

    def _build_val_view(
        self,
        image: Image.Image,
        mask: Image.Image,
        image_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_size == self.image_size:
            val_transform = self.val_transform
            val_mask_transform = self.val_mask_transform
        else:
            val_transform = self._build_val_image_transform(image_size)
            val_mask_transform = self._build_val_mask_transform(image_size)

        image_tensor = val_transform(image)
        mask_tensor = (val_mask_transform(mask) > 0.5).float()
        return image_tensor, self.mask_dilation(mask_tensor)
