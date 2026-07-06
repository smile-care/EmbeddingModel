"""Inspect SupConDataset samples and training augmentations.

Example:
  python scripts_training/inspect_supcon_dataset.py \
    --config configs/supcon_config.yaml \
    --copy-paste-prob 1.0
"""
import argparse
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding_model.supcon.datasets import SupConDataset
from src.embedding_model.utils.config_loader import load_config


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SupConDataset augmentation outputs.")
    parser.add_argument("--config", default="configs/supcon_config.yaml", help="SupCon config path.")
    parser.add_argument("--data-config", default=None, help="Override data config path.")
    parser.add_argument("--save-dir", default=None, help="Optional directory to save displayed grids.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split to inspect.")
    parser.add_argument("--scene", default=None, help="Only inspect scenes whose name contains this text.")
    parser.add_argument("--num-samples", type=int, default=32, help="Total samples to inspect.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--image-size", type=int, default=None, help="Override visualization/training image size.")
    parser.add_argument("--window-name", default="SupCon Dataset Inspector", help="cv2 window name.")
    parser.add_argument(
        "--copy-paste-prob",
        type=float,
        default=None,
        help="Override copy_paste.prob for inspection, e.g. 1.0 to force distractors.",
    )
    parser.add_argument("--include-multiscale-masks", action="store_true", help="Append 4x/8x/16x/32x masks.")
    return parser.parse_args()


def resolve_image_size(value, override: int | None) -> int:
    if override is not None:
        return override
    if isinstance(value, int):
        return value
    if isinstance(value, list) and value:
        return int(max(value))
    return 224


def clone_copy_paste_config(config: dict, override_prob: float | None) -> dict:
    copied = dict(config or {"enabled": False})
    if override_prob is not None:
        copied["enabled"] = True
        copied["prob"] = float(override_prob)
    return copied


def tensor_image_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    image = torch.clamp(tensor * STD + MEAN, 0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def tensor_mask_to_pil(mask: torch.Tensor) -> Image.Image:
    mask = mask.detach().cpu()
    if mask.ndim == 3:
        mask = mask[0]
    array = (torch.clamp(mask, 0.0, 1.0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def load_source_pair(sample: dict, size: int) -> tuple[Image.Image, Image.Image]:
    image = Image.open(sample["image_path"]).convert("RGB").resize((size, size), Image.BILINEAR)
    mask = Image.open(sample["mask_path"]).convert("L").resize((size, size), Image.NEAREST)
    return image, mask


def overlay_mask(image: Image.Image, mask: Image.Image, color=(255, 32, 32), alpha: int = 110) -> Image.Image:
    image = image.convert("RGBA")
    mask_np = np.array(mask) > 0
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_np = np.array(overlay)
    overlay_np[mask_np] = (*color, alpha)
    return Image.alpha_composite(image, Image.fromarray(overlay_np, mode="RGBA")).convert("RGB")


def label_tile(tile: Image.Image, text: str) -> Image.Image:
    out = tile.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 4
    draw.rectangle(
        (0, 0, bbox[2] + 2 * pad, bbox[3] + 2 * pad),
        fill=(0, 0, 0),
    )
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def mask_pyramid(mask: torch.Tensor, base_size: int) -> list[tuple[str, Image.Image]]:
    levels = []
    for stride in (4, 8, 16, 32):
        resized = F.interpolate(mask.unsqueeze(0), scale_factor=1 / stride, mode="nearest").squeeze(0)
        mask_img = tensor_mask_to_pil(resized).resize((base_size, base_size), Image.NEAREST).convert("RGB")
        levels.append((f"mask /{stride}", mask_img))
    return levels


def make_grid(tiles: Iterable[tuple[str, Image.Image]], columns: int = 3, pad: int = 8) -> Image.Image:
    tiles = [(title, tile.convert("RGB")) for title, tile in tiles]
    if not tiles:
        raise ValueError("tiles must not be empty")
    tile_w, tile_h = tiles[0][1].size
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w + (columns + 1) * pad, rows * tile_h + (rows + 1) * pad), "white")
    for idx, (title, tile) in enumerate(tiles):
        row, col = divmod(idx, columns)
        x = pad + col * (tile_w + pad)
        y = pad + row * (tile_h + pad)
        canvas.paste(label_tile(tile, title), (x, y))
    return canvas


def show_grid(window_name: str, grid: Image.Image) -> int:
    bgr = cv2.cvtColor(np.array(grid), cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, bgr)
    return cv2.waitKey(0) & 0xFF


def build_dataset(scene: dict, split: str, image_size: int, mask_dilation: dict, copy_paste: dict) -> SupConDataset:
    return SupConDataset(
        root=scene["root"],
        split=split,
        image_size=image_size,
        name=scene.get("name", scene["root"]),
        mask_dilation_config=mask_dilation,
        copy_paste_config=copy_paste if split == "train" else {"enabled": False},
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    supcon_config = load_config(args.config)["supcon"]
    data_config_path = args.data_config or supcon_config["data"]["data_config_path"]
    data_config = load_config(data_config_path)

    image_size = resolve_image_size(supcon_config["data"].get("image_size", 224), args.image_size)
    mask_dilation = supcon_config["data"].get("mask_dilation", {"enabled": False})
    copy_paste = clone_copy_paste_config(
        supcon_config["data"].get("copy_paste", {"enabled": False}),
        args.copy_paste_prob,
    )

    scenes = data_config["scenes"].get(args.split, [])
    if args.scene:
        scenes = [s for s in scenes if args.scene in s.get("name", s["root"])]
    if not scenes:
        raise ValueError(f"No scenes found for split={args.split!r}, scene filter={args.scene!r}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    inspected = 0

    print("Controls: n/space/enter = next, q/esc = quit")

    for scene in scenes:
        if inspected >= args.num_samples:
            break
        scene_name = scene.get("name", scene["root"])
        try:
            dataset = build_dataset(scene, args.split, image_size, mask_dilation, copy_paste)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] {scene_name}: {exc}")
            continue
        if len(dataset) == 0:
            print(f"[skip] {scene_name}: empty dataset")
            continue

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for idx in indices:
            if inspected >= args.num_samples:
                break
            item = dataset[idx]
            source_image, source_mask = load_source_pair(dataset.samples[idx], image_size)
            view1_image = tensor_image_to_pil(item["view1_image"])
            view2_image = tensor_image_to_pil(item["view2_image"])
            view1_mask = tensor_mask_to_pil(item["view1_mask"])
            view2_mask = tensor_mask_to_pil(item["view2_mask"])

            tiles = [
                ("source", source_image),
                ("source mask", source_mask.convert("RGB")),
                ("source overlay", overlay_mask(source_image, source_mask)),
                ("view1", view1_image),
                ("view1 mask", view1_mask.convert("RGB")),
                ("view1 overlay", overlay_mask(view1_image, view1_mask)),
                ("view2", view2_image),
                ("view2 mask", view2_mask.convert("RGB")),
                ("view2 overlay", overlay_mask(view2_image, view2_mask)),
            ]
            if args.include_multiscale_masks:
                tiles.extend(mask_pyramid(item["view1_mask"], image_size))

            grid = make_grid(tiles, columns=3)
            title = (
                f"[{inspected + 1}/{args.num_samples}] "
                f"scene={scene_name}, label={item['label_name']}, "
                f"file={Path(item['image_path']).name}"
            )
            print(title)

            if save_dir is not None:
                safe_scene = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in scene_name)[:80]
                safe_label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in item["label_name"])[:40]
                out_path = save_dir / f"{inspected:04d}_{safe_scene}_{safe_label}.png"
                grid.save(out_path)
                print(f"saved: {out_path}")

            key = show_grid(args.window_name, grid)
            inspected += 1
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"Stopped. Inspected {inspected} samples.")
                return

    cv2.destroyAllWindows()
    print(f"Done. Inspected {inspected} samples.")


if __name__ == "__main__":
    main()
