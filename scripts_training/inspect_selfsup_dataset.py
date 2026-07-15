"""Save a visual audit grid for the independent self-supervised augmentations."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding_model.selfsup.config import load_selfsup_config
from src.embedding_model.selfsup.datasets import SelfSupervisedDataset


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 selfsup 双视图增强和 Mask")
    parser.add_argument("--config", default="configs/selfsup_config.yaml")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="/tmp/selfsup_augmentation_audit.png")
    return parser.parse_args()


def data_sources(path: str) -> list[dict]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["scenes"]["train"]


def image_array(tensor: torch.Tensor) -> np.ndarray:
    image = (tensor.cpu() * STD + MEAN).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def draw_view(axis, image: torch.Tensor, mask: torch.Tensor, title: str) -> None:
    array = image_array(image)
    axis.imshow(array)
    axis.imshow(mask.squeeze(0).cpu().numpy(), cmap="Reds", alpha=0.25, vmin=0, vmax=1)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_selfsup_config(args.config)
    dataset = SelfSupervisedDataset(
        data_sources(config["data"]["data_config_path"]),
        project_root=ROOT,
        image_size=int(config["data"]["image_size"]),
        augmentation_config=config["data"].get("augmentation", {}),
        training=True,
    )
    print(
        f"dataset samples={len(dataset)}, deduplicated={dataset.duplicate_count}, "
        f"missing_masks={dataset.missing_mask_count}"
    )
    sample_count = min(max(1, args.samples), len(dataset))
    indices = random.sample(range(len(dataset)), sample_count)
    figure, axes = plt.subplots(sample_count, 2, figsize=(8, 4 * sample_count), squeeze=False)
    for row, index in enumerate(indices):
        sample = dataset[index]
        draw_view(axes[row, 0], sample["view1_image"], sample["view1_mask"], "view 1")
        draw_view(axes[row, 1], sample["view2_image"], sample["view2_mask"], "view 2")
        axes[row, 0].set_ylabel(
            f"{sample['source_name']}\n{sample['label_name']}",
            fontsize=8,
        )
    figure.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"增强审计图已保存: {output_path}")


if __name__ == "__main__":
    main()
