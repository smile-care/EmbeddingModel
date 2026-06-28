"""
基于 manifest（CSV 或内联 samples）的三元组训练数据集。

每条样本统一为 [image, mask, label]，输出格式与 SupConDataset 一致，
因此训练逻辑无需改动。该数据集为通用 DL 数据加载代码，由 data_cluster
平台与其他训练入口共同使用。
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from embedding_model.supcon.datasets.supcon_dataset import (IndexWithScale, MaskSoftDilation,
                                                            TwoViewAugmentation)


class ManifestTripletDataset(torch.utils.data.Dataset):
    """通用三元组数据集：每条样本必须包含 image/mask/label。"""

    def __init__(
        self,
        data_config: dict[str, Any],
        split: str = "train",
        image_size: int | list[int] = 224,
    ) -> None:
        self.config = data_config
        self.split = split

        if isinstance(image_size, int):
            self.image_sizes = [image_size]
        elif isinstance(image_size, list):
            self.image_sizes = image_size
        else:
            raise ValueError(f"image_size必须是int或List[int]，当前为{type(image_size)}")
        self.image_size = self.image_sizes[0]

        self.default_similarity = float(self.config.get("default_similarity", 0.0))
        self.custom_similarity = list(self.config.get("custom_similarity", []))
        self.samples = self._load_triplets()
        if not self.samples:
            raise ValueError(f"{self.split} split 没有可用样本，请检查 crop manifest")

        self.categories = sorted({str(sample["label"]) for sample in self.samples})
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}
        for sample in self.samples:
            sample["label_idx"] = self.cat2idx[str(sample["label"])]

        self.similarity_matrix = self._build_similarity_matrix()
        mask_dilation_cfg = self.config.get("mask_dilation")
        if not isinstance(mask_dilation_cfg, dict):
            mask_dilation_cfg = {"enabled": True, "method": "fast_pool", "fast_gamma": 0.8}

        self.augmentation = (
            TwoViewAugmentation(
                {
                    "image_size": self.image_size,
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
                        "brightness": 0.2,
                        "contrast": 0.2,
                        "saturation": 0.1,
                        "hue": 0.05,
                    },
                    "mask_dilation": mask_dilation_cfg,
                }
            )
            if split == "train"
            else None
        )
        self.mask_dilation = MaskSoftDilation(mask_dilation_cfg)
        if split != "train":
            self.val_transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            self.val_mask_transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ]
            )

    def _read_rows(self) -> list[dict[str, Any]]:
        inline_samples = self.config.get("samples")
        manifest_path = self.config.get("manifest_path")

        if isinstance(inline_samples, list):
            return [dict(item) for item in inline_samples]
        if manifest_path:
            path = Path(str(manifest_path))
            if not path.exists():
                raise FileNotFoundError(f"训练索引清单不存在: {path}")
            with path.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        raise ValueError("ManifestTripletDataset 需要 manifest_path 或 samples")

    def _load_triplets(self) -> list[dict[str, Any]]:
        rows = self._read_rows()
        samples: list[dict[str, Any]] = []
        missing_rows: list[str] = []

        for row in rows:
            sample_split = str(row.get("split", "train")).strip().lower() or "train"
            if sample_split != self.split:
                continue

            image_path = Path(str(row.get("image_path", "")).strip())
            mask_path_raw = str(row.get("mask_path", "")).strip()
            label = str(row.get("label") or row.get("category_name") or "").strip()
            mask_path = Path(mask_path_raw) if mask_path_raw else None

            if not label:
                missing_rows.append(f"missing label for image={image_path}")
                continue
            if not image_path.exists():
                missing_rows.append(f"image not found: {image_path}")
                continue
            if mask_path is None or not mask_path.exists():
                missing_rows.append(f"mask not found: {mask_path}")
                continue

            samples.append(
                {
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "label": label,
                    "crop_image_id": row.get("crop_image_id"),
                    "source_image_id": row.get("source_image_id"),
                }
            )

        if not samples and missing_rows:
            preview = "; ".join(missing_rows[:5])
            raise ValueError(f"{self.split} split 没有有效样本: {preview}")
        return samples

    def _build_similarity_matrix(self) -> np.ndarray:
        num_categories = len(self.categories)
        similarity_matrix = np.full((num_categories, num_categories), self.default_similarity, dtype=float)
        np.fill_diagonal(similarity_matrix, 1.0)
        for item in self.custom_similarity:
            categories_list = item.get("list", [])
            sim = item.get("similarity", self.default_similarity)
            for i, cat1 in enumerate(categories_list):
                for cat2 in categories_list[i + 1:]:
                    if cat1 not in self.cat2idx or cat2 not in self.cat2idx:
                        continue
                    idx1 = self.cat2idx[cat1]
                    idx2 = self.cat2idx[cat2]
                    similarity_matrix[idx1, idx2] = sim
                    similarity_matrix[idx2, idx1] = sim
        return similarity_matrix

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx, current_image_size: Optional[int] = None) -> dict[str, Any]:
        if isinstance(idx, IndexWithScale):
            actual_idx = idx.idx
            current_image_size = idx.image_size
        else:
            actual_idx = idx

        sample = self.samples[actual_idx]
        if current_image_size is not None:
            image_size = current_image_size
        elif len(self.image_sizes) == 1:
            image_size = self.image_sizes[0]
        else:
            image_size = random.choice(self.image_sizes)

        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")

        if self.augmentation is not None:
            view1_image, view1_mask, view2_image, view2_mask = self.augmentation(
                image, mask, image_size=image_size
            )
        else:
            if image_size != self.image_size:
                val_transform = transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
                val_mask_transform = transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                    ]
                )
            else:
                val_transform = self.val_transform
                val_mask_transform = self.val_mask_transform

            view1_image = val_transform(image)
            view1_mask = (val_mask_transform(mask) > 0.5).float()
            view1_mask = self.mask_dilation(view1_mask)
            view2_image = view1_image.clone()
            view2_mask = view1_mask.clone()

        return {
            "view1_image": view1_image,
            "view1_mask": view1_mask,
            "view2_image": view2_image,
            "view2_mask": view2_mask,
            "label": sample["label_idx"],
            "label_name": sample["label"],
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"],
            "crop_image_id": sample.get("crop_image_id"),
            "source_image_id": sample.get("source_image_id"),
        }

    def get_similarity_matrix(self) -> np.ndarray:
        return self.similarity_matrix.copy()


# Backwards-compatible alias (platform historically referenced this name).
DataClusterTripletDataset = ManifestTripletDataset
