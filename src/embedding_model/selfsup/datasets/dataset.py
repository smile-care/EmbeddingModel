"""A flat, uniformly sampled dataset for self-supervised defect learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentations import MaskPreservingTwoViewTransform, build_eval_view


class SelfSupervisedDataset(Dataset):
    """Merge every configured source into one sample pool without label supervision."""

    def __init__(
        self,
        source_configs: Iterable[dict[str, Any]],
        *,
        project_root: str | Path,
        image_size: int,
        augmentation_config: dict[str, Any] | None = None,
        training: bool = True,
    ) -> None:
        self.project_root = Path(project_root)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.transform = MaskPreservingTwoViewTransform(
            image_size=self.image_size,
            config=augmentation_config,
        )
        self.samples, self.duplicate_count, self.missing_mask_count = self._scan_sources(source_configs)
        if not self.samples:
            raise ValueError("没有找到任何具有对应 _mask.png 的自监督训练样本")

    def _scan_sources(
        self,
        source_configs: Iterable[dict[str, Any]],
    ) -> tuple[list[dict[str, str]], int, int]:
        samples: list[dict[str, str]] = []
        seen_paths: set[Path] = set()
        duplicate_count = 0
        missing_mask_count = 0

        for source_config in source_configs:
            if "root" not in source_config:
                raise ValueError(f"数据 source 缺少 root: {source_config}")
            source_name = str(source_config.get("name", source_config["root"]))
            source_root = Path(source_config["root"])
            if not source_root.is_absolute():
                source_root = self.project_root / source_root
            if not source_root.is_dir():
                raise FileNotFoundError(f"数据目录不存在: {source_root}")

            for image_path in sorted(source_root.rglob("*.png")):
                if image_path.name.endswith("_mask.png"):
                    continue
                resolved_path = image_path.resolve()
                if resolved_path in seen_paths:
                    duplicate_count += 1
                    continue
                mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
                if not mask_path.is_file():
                    missing_mask_count += 1
                    continue

                seen_paths.add(resolved_path)
                samples.append(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "label_name": image_path.parent.name,
                        "source_name": source_name,
                    }
                )
        return samples, duplicate_count, missing_mask_count

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as image_file:
            image = image_file.convert("RGB")
        with Image.open(sample["mask_path"]) as mask_file:
            mask = mask_file.convert("L")

        if self.training:
            image1, mask1, image2, mask2 = self.transform(image, mask)
        else:
            image1, mask1 = build_eval_view(image, mask, self.image_size)
            image2, mask2 = image1.clone(), mask1.clone()

        return {
            "view1_image": image1,
            "view1_mask": mask1,
            "view2_image": image2,
            "view2_mask": mask2,
            "label_name": sample["label_name"],
            "source_name": sample["source_name"],
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"],
        }


def selfsup_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensors while preserving diagnostic metadata as strings."""
    tensor_keys = {"view1_image", "view1_mask", "view2_image", "view2_mask"}
    return {
        key: torch.stack([item[key] for item in batch]) if key in tensor_keys else [item[key] for item in batch]
        for key in batch[0]
    }
