"""Image/mask augmentations for SupCon training."""
import copy
import random
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision import transforms
import torchvision.transforms.functional as TF


DEFAULT_TRAIN_AUGMENTATION: dict[str, Any] = {
    "horizontal_flip": {"enabled": True, "prob": 0.5},
    "affine": {
        "enabled": True,
        "degrees": 15,
        "translate": (0.2, 0.2),
        "scale": (0.8, 1.2),
        "shear": 15,
        "fill": 0,
    },
    "color_jitter": {
        "enabled": True,
        "brightness": 0.05,
        "contrast": 0.05,
        "saturation": 0.02,
        "hue": 0.0,
    },
}


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_train_augmentation_config(
    train_augmentation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not train_augmentation:
        return copy.deepcopy(DEFAULT_TRAIN_AUGMENTATION)
    return deep_merge_dict(DEFAULT_TRAIN_AUGMENTATION, train_augmentation)


def build_two_view_augmentation_config(
    *,
    image_size: int,
    train_augmentation: Optional[dict[str, Any]] = None,
    mask_dilation_config: Optional[dict[str, Any]] = None,
    copy_paste_config: Optional[dict[str, Any]] = None,
    copy_paste_samples: Optional[list] = None,
) -> dict[str, Any]:
    aug = resolve_train_augmentation_config(train_augmentation)
    return {
        "image_size": image_size,
        **aug,
        "mask_dilation": mask_dilation_config or {"enabled": False},
        "copy_paste": copy_paste_config or {"enabled": False},
        "copy_paste_samples": copy_paste_samples or [],
    }


class MaskSoftDilation:
    """Mask软膨胀处理类"""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        self.enabled = config.get('enabled', True)

    def __call__(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return mask_tensor

        if mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[0:1]

        mask_np = mask_tensor.squeeze(0).cpu().numpy()
        binary_mask = (mask_np > 0.5).astype(np.float32)
        if binary_mask.sum() == 0:
            return mask_tensor

        foreground_area = binary_mask.sum()
        equivalent_radius = np.sqrt(foreground_area / np.pi)
        dilation_radius = min(30, max(10, int(equivalent_radius)))
        distance = ndimage.distance_transform_edt(1 - binary_mask)

        soft_mask = np.zeros_like(binary_mask, dtype=np.float32)
        dilation_region = (distance > 0) & (distance <= dilation_radius)
        if dilation_region.any():
            soft_mask[dilation_region] = 1.0 - distance[dilation_region] / dilation_radius
        soft_mask[binary_mask > 0.5] = 1.0

        return torch.from_numpy(soft_mask).unsqueeze(0).to(mask_tensor.device)


class CopyPasteDistractor:
    """Paste extra defect regions outside the current mask as background distractors."""

    def __init__(self, config: Optional[Dict], samples: Optional[List[Dict]] = None):
        config = config or {}
        self.enabled = bool(config.get('enabled', False))
        self.samples = samples or []
        self.prob = float(config.get('prob', 0.0))
        self.max_pastes = int(config.get('max_pastes', 1))
        self.avoid_mask_dilation = int(config.get('avoid_mask_dilation', 8))
        self.max_attempts = int(config.get('max_attempts', 10))
        self.cache_size = int(config.get('cache_size', 0))
        self.different_label_prob = float(config.get('different_label_prob', 0.7))
        self.opacity_range = tuple(config.get('opacity_range', [0.85, 1.0]))
        self.scale_range = tuple(config.get('scale_range', [0.7, 1.1]))
        self.max_size_ratio = float(config.get('max_size_ratio', 0.45))
        self.indices_by_label = self._build_label_index(self.samples)
        self.all_indices = list(range(len(self.samples)))
        self.indices_except_label = self._build_except_label_index()
        self._image_cache = OrderedDict()
        self._mask_cache = OrderedDict()

    @staticmethod
    def _build_label_index(samples: List[Dict]) -> Dict[str, List[int]]:
        indices_by_label = defaultdict(list)
        for idx, sample in enumerate(samples):
            indices_by_label[sample.get('label')].append(idx)
        return dict(indices_by_label)

    def _build_except_label_index(self) -> Dict[str, List[int]]:
        indices_except_label = {}
        for label, indices in self.indices_by_label.items():
            same_label_indices = set(indices)
            indices_except_label[label] = [
                idx for idx in self.all_indices if idx not in same_label_indices
            ]
        return indices_except_label

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        current_sample: Optional[Dict] = None,
    ) -> Image.Image:
        if not self.enabled or not self.samples or random.random() >= self.prob:
            return image

        out = image.copy()
        paste_count = random.randint(1, max(1, self.max_pastes))
        for _ in range(paste_count):
            donor = self._sample_donor(current_sample)
            if donor is not None:
                out = self._paste_one(out, mask, donor)
        return out

    def _sample_donor(self, current_sample: Optional[Dict]) -> Optional[Dict]:
        current_path = current_sample.get('image_path') if current_sample else None
        current_label = current_sample.get('label') if current_sample else None

        if not self.all_indices:
            return None

        candidate_indices = self.all_indices
        if current_label is not None and random.random() < self.different_label_prob:
            diff_indices = self.indices_except_label.get(current_label, [])
            if diff_indices:
                candidate_indices = diff_indices

        for _ in range(self.max_attempts):
            donor = self.samples[random.choice(candidate_indices)]
            if donor.get('image_path') != current_path:
                return donor
        return None

    def _paste_one(self, image: Image.Image, target_mask: Image.Image, donor: Dict) -> Image.Image:
        try:
            donor_image = self._load_cached(donor['image_path'], 'RGB', self._image_cache)
            donor_mask = self._load_cached(donor['mask_path'], 'L', self._mask_cache)
        except (OSError, KeyError):
            return image

        donor_crop = self._extract_donor_crop(donor_image, donor_mask, image.size)
        if donor_crop is None:
            return image
        patch_img, patch_alpha = donor_crop

        protected = self._protected_mask(target_mask, image.size)
        image_w, image_h = image.size
        patch_w, patch_h = patch_img.size
        if patch_w <= 0 or patch_h <= 0 or patch_w > image_w or patch_h > image_h:
            return image

        alpha_np = np.array(patch_alpha) > 0
        if not alpha_np.any():
            return image

        out = image.copy()
        for _ in range(self.max_attempts):
            x = random.randint(0, image_w - patch_w)
            y = random.randint(0, image_h - patch_h)
            protected_region = protected[y:y + patch_h, x:x + patch_w]
            if protected_region.shape != alpha_np.shape or np.any(protected_region & alpha_np):
                continue
            out.paste(patch_img, (x, y), patch_alpha)
            return out
        return image

    def _load_cached(self, path: str, mode: str, cache: OrderedDict) -> Image.Image:
        if self.cache_size <= 0:
            return Image.open(path).convert(mode)

        cached = cache.get(path)
        if cached is not None:
            cache.move_to_end(path)
            return cached.copy()

        image = Image.open(path).convert(mode)
        cache[path] = image.copy()
        cache.move_to_end(path)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)
        return image

    def _extract_donor_crop(
        self,
        donor_image: Image.Image,
        donor_mask: Image.Image,
        target_size: Tuple[int, int],
    ) -> Optional[Tuple[Image.Image, Image.Image]]:
        mask_np = np.array(donor_mask) > 127
        if not mask_np.any():
            return None

        ys, xs = np.where(mask_np)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        patch_img = donor_image.crop((x0, y0, x1, y1))
        patch_mask = donor_mask.crop((x0, y0, x1, y1))

        target_w, target_h = target_size
        max_w = max(1, int(target_w * self.max_size_ratio))
        max_h = max(1, int(target_h * self.max_size_ratio))
        scale = random.uniform(float(self.scale_range[0]), float(self.scale_range[1]))
        scale = min(scale, max_w / max(1, patch_img.width), max_h / max(1, patch_img.height))
        if scale <= 0:
            return None

        new_size = (
            max(1, int(round(patch_img.width * scale))),
            max(1, int(round(patch_img.height * scale))),
        )
        if new_size != patch_img.size:
            patch_img = patch_img.resize(new_size, Image.BILINEAR)
            patch_mask = patch_mask.resize(new_size, Image.NEAREST)

        opacity = random.uniform(float(self.opacity_range[0]), float(self.opacity_range[1]))
        alpha = np.array(patch_mask).astype(np.float32)
        alpha = np.clip(alpha * opacity, 0, 255).astype(np.uint8)
        return patch_img, Image.fromarray(alpha, mode='L')

    def _protected_mask(self, mask: Image.Image, size: Tuple[int, int]) -> np.ndarray:
        mask_np = np.array(mask.resize(size, Image.NEAREST)) > 127
        if self.avoid_mask_dilation > 0 and mask_np.any():
            structure = np.ones(
                (2 * self.avoid_mask_dilation + 1, 2 * self.avoid_mask_dilation + 1),
                dtype=bool,
            )
            mask_np = ndimage.binary_dilation(mask_np, structure=structure)
        return mask_np


class TwoViewAugmentation:
    """双视图数据增强（同时处理图像和mask）"""

    def __init__(self, config: Optional[Dict]):
        config = config or {}
        self.image_size = config.get('image_size', 224)
        self.config = config

        hflip_cfg = config.get('horizontal_flip', {}) or {}
        self.flip_enabled = bool(hflip_cfg.get('enabled', True))
        self.flip_prob = float(hflip_cfg.get('prob', 0.5))

        affine_cfg = config.get('affine', {}) or {}
        self.affine_enabled = bool(affine_cfg.get('enabled', False))
        self.affine_degrees = self._as_symmetric_range(affine_cfg.get('degrees', 0))
        translate = affine_cfg.get('translate', (0.2, 0.2))
        self.affine_translate = tuple(float(t) for t in translate) if translate else None
        scale = affine_cfg.get('scale', (0.8, 1.2))
        self.affine_scale = tuple(float(s) for s in scale) if scale else None
        self.affine_shear = self._as_symmetric_range(affine_cfg.get('shear', 0))
        self.affine_fill = affine_cfg.get('fill', 0)

        color_jitter_cfg = config.get('color_jitter', {}) or {}
        if color_jitter_cfg.get('enabled', False):
            self.color_jitter = transforms.ColorJitter(
                brightness=color_jitter_cfg.get('brightness', 0.1),
                contrast=color_jitter_cfg.get('contrast', 0.1),
                saturation=color_jitter_cfg.get('saturation', 0.1),
                hue=color_jitter_cfg.get('hue', 0.05),
            )
        else:
            self.color_jitter = None

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.mask_dilation = MaskSoftDilation(config.get('mask_dilation', {}))
        self.copy_paste = CopyPasteDistractor(
            config.get('copy_paste', {}),
            samples=config.get('copy_paste_samples', []),
        )

    @staticmethod
    def _as_symmetric_range(value) -> Optional[Tuple[float, float]]:
        """把标量 v 转成 (-v, v)；已是区间则原样返回；0/None 表示禁用该项。"""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return tuple(float(x) for x in value)
        v = float(value)
        if v == 0.0:
            return None
        return (-v, v)

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        image_size: Optional[int] = None,
        current_sample: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        size = image_size if image_size is not None else self.image_size

        view1_source = self.copy_paste(image, mask, current_sample=current_sample)
        view2_source = self.copy_paste(image, mask, current_sample=current_sample)

        view1_image, view1_mask = self._apply_augmentation_with_mask(view1_source, mask, size)
        view2_image, view2_mask = self._apply_augmentation_with_mask(view2_source, mask, size)

        return (
            self.normalize(view1_image),
            view1_mask,
            self.normalize(view2_image),
            view2_mask,
        )

    def _apply_augmentation_with_mask(
        self,
        image: Image.Image,
        mask: Image.Image,
        size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 image/mask 施加同一套几何增强（几何参数只采样一次）。

        与旧实现语义保持一致：resize 用双线性、仿射用最近邻，颜色抖动仅作用于
        image，mask 最终二值化 (>0.5) 后再做软膨胀。
        """
        # 几何：resize（image/mask 均双线性，沿用旧默认行为）
        image = TF.resize(image, [size, size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [size, size], interpolation=TF.InterpolationMode.BILINEAR)

        # 几何：水平翻转（采样一次，image/mask 同步）
        if self.flip_enabled and random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 几何：仿射（参数只采样一次，image/mask 用同一参数，均最近邻）
        if self.affine_enabled:
            angle, translations, scale, shear = transforms.RandomAffine.get_params(
                self.affine_degrees if self.affine_degrees is not None else (0.0, 0.0),
                self.affine_translate,
                self.affine_scale,
                self.affine_shear,
                [size, size],
            )
            image = TF.affine(
                image, angle=angle, translate=translations, scale=scale, shear=shear,
                interpolation=TF.InterpolationMode.NEAREST, fill=self.affine_fill,
            )
            mask = TF.affine(
                mask, angle=angle, translate=translations, scale=scale, shear=shear,
                interpolation=TF.InterpolationMode.NEAREST, fill=0,
            )

        # 颜色：仅作用于 image
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image_tensor = TF.to_tensor(image)
        mask_tensor = TF.to_tensor(mask)

        if mask_tensor.shape[1:] != image_tensor.shape[1:]:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=image_tensor.shape[1:],
                mode='nearest',
            ).squeeze(0)

        mask_tensor = (mask_tensor[:1] > 0.5).float()
        return image_tensor, self.mask_dilation(mask_tensor)
