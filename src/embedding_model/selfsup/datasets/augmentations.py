"""Mask-preserving two-view augmentations for industrial defect crops."""

from __future__ import annotations

import random
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MaskPreservingTwoViewTransform:
    """Create two independent views while keeping the annotated defect visible."""

    def __init__(self, image_size: int, config: dict[str, Any] | None = None):
        self.image_size = int(image_size)
        self.config = config or {}
        self.horizontal_flip_prob = float(self.config.get("horizontal_flip_prob", 0.5))
        self.degrees = float(self.config.get("degrees", 10.0))
        self.translate = tuple(self.config.get("translate", [0.1, 0.1]))
        self.scale = tuple(self.config.get("scale", [0.9, 1.1]))
        self.shear = float(self.config.get("shear", 5.0))
        self.min_mask_pixels = int(self.config.get("min_mask_pixels", 4))
        self.max_geometry_attempts = int(self.config.get("max_geometry_attempts", 5))
        self.blur_prob = float(self.config.get("gaussian_blur_prob", 0.1))
        self.blur_kernel_size = int(self.config.get("gaussian_blur_kernel_size", 5))
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1

        jitter_cfg = self.config.get("color_jitter", {})
        self.color_jitter = transforms.ColorJitter(
            brightness=float(jitter_cfg.get("brightness", 0.08)),
            contrast=float(jitter_cfg.get("contrast", 0.08)),
            saturation=float(jitter_cfg.get("saturation", 0.05)),
            hue=float(jitter_cfg.get("hue", 0.02)),
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image1, mask1 = self._make_view(image, mask)
        image2, mask2 = self._make_view(image, mask)
        return image1, mask1, image2, mask2

    def _make_view(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        transformed_image: Image.Image | None = None
        transformed_mask: Image.Image | None = None

        for _ in range(max(1, self.max_geometry_attempts)):
            candidate_image, candidate_mask = self._geometric_transform(image, mask)
            candidate_mask_tensor = self._mask_to_tensor(candidate_mask)
            if int(candidate_mask_tensor.sum().item()) >= self.min_mask_pixels:
                transformed_image = candidate_image
                transformed_mask = candidate_mask
                break

        if transformed_image is None or transformed_mask is None:
            transformed_image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            transformed_mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )

        transformed_image = self.color_jitter(transformed_image)
        if random.random() < self.blur_prob:
            transformed_image = TF.gaussian_blur(transformed_image, self.blur_kernel_size)

        image_tensor = TF.to_tensor(transformed_image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        return image_tensor, self._mask_to_tensor(transformed_mask)

    def _geometric_transform(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle, translations, scale, shear = transforms.RandomAffine.get_params(
            degrees=[-self.degrees, self.degrees],
            translate=list(self.translate),
            scale_ranges=list(self.scale),
            shears=[-self.shear, self.shear],
            img_size=list(image.size),
        )
        image = TF.affine(
            image,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        return image, mask

    @staticmethod
    def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
        return (TF.to_tensor(mask) > 0.5).float()


def build_eval_view(image: Image.Image, mask: Image.Image, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic view for representation evaluation."""
    image = TF.resize(
        image,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    mask = TF.resize(mask, [image_size, image_size], interpolation=InterpolationMode.NEAREST)
    image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
    return image_tensor, (TF.to_tensor(mask) > 0.5).float()
