"""
从图像和多边形标注（归一化 0-1）提取缺陷 patch。

每个独立的非 subtract 标注（label）单独生成一张 crop 图及其 mask，
subtract 标注用于在重叠区域挖洞。此模块为 DL 数据预处理代码，供
data_cluster 平台与离线脚本共同调用。
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def extract_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """从 mask 提取外接矩形 (x_min, y_min, x_max, y_max) 或 None。"""
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def polygon_to_mask(polygon: List[List[float]], img_h: int, img_w: int) -> np.ndarray:
    """将归一化的多边形坐标 (0-1) 转换为像素 mask (H, W) uint8。"""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(polygon) < 3:
        return mask
    points = np.array(polygon, dtype=np.float32)
    points[:, 0] *= img_w
    points[:, 1] *= img_h
    points = points.astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def extract_instances_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """从 mask 中提取所有独立的连通区域 instance。"""
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    instances = []
    for label_id in range(1, num_labels):
        instance_mask = (labels == label_id).astype(np.uint8) * 255
        instances.append(instance_mask)
    return instances


def compute_crop_region(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    crop_sizes: List[int] = [96, 160, 224],
) -> Tuple[int, int, int, int]:
    """根据自适应策略计算正方形 crop 区域 (x_min, y_min, x_max, y_max)。"""
    x_min, y_min, x_max, y_max = bbox
    img_h, img_w = image_shape

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_max = max(bbox_w, bbox_h)

    min_required_crop_size = int(bbox_max * 2)

    if bbox_max < crop_sizes[0]:
        crop_size = max(crop_sizes[0], min_required_crop_size)
        use_expand = False
    elif bbox_max < crop_sizes[1]:
        crop_size = max(crop_sizes[1], min_required_crop_size)
        use_expand = False
    elif bbox_max < crop_sizes[2]:
        crop_size = max(crop_sizes[2], min_required_crop_size)
        use_expand = False
    else:
        crop_size = None
        use_expand = True

    if use_expand:
        min_padding = bbox_max / 2
        expand_w = int(min_padding)
        expand_h = int(min_padding)

        expanded_x_min = max(0, x_min - expand_w)
        expanded_y_min = max(0, y_min - expand_h)
        expanded_x_max = min(img_w, x_max + expand_w)
        expanded_y_max = min(img_h, y_max + expand_h)

        expanded_w = expanded_x_max - expanded_x_min
        expanded_h = expanded_y_max - expanded_y_min

        crop_size_final = max(expanded_w, expanded_h, min_required_crop_size)

        bbox_center_x = (x_min + x_max) // 2
        bbox_center_y = (y_min + y_max) // 2

        half_size = crop_size_final // 2
        crop_x_min = max(0, bbox_center_x - half_size)
        crop_y_min = max(0, bbox_center_y - half_size)
        crop_x_max = min(img_w, crop_x_min + crop_size_final)
        crop_y_max = min(img_h, crop_y_min + crop_size_final)

        final_w = crop_x_max - crop_x_min
        final_h = crop_y_max - crop_y_min
        final_crop_size = min(final_w, final_h)

        if final_w != final_h:
            center_x = (crop_x_min + crop_x_max) // 2
            center_y = (crop_y_min + crop_y_max) // 2
            half_final = final_crop_size // 2
            crop_x_min = max(0, center_x - half_final)
            crop_y_min = max(0, center_y - half_final)
            crop_x_max = min(img_w, crop_x_min + final_crop_size)
            crop_y_max = min(img_h, crop_y_min + final_crop_size)
    else:
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        half_size = crop_size // 2
        crop_x_min = max(0, center_x - half_size)
        crop_y_min = max(0, center_y - half_size)
        crop_x_max = min(img_w, crop_x_min + crop_size)
        crop_y_max = min(img_h, crop_y_min + crop_size)

        if crop_x_max - crop_x_min < crop_size:
            crop_x_min = max(0, crop_x_max - crop_size)
        if crop_y_max - crop_y_min < crop_size:
            crop_y_min = max(0, crop_y_max - crop_size)

        final_w = crop_x_max - crop_x_min
        final_h = crop_y_max - crop_y_min
        final_crop_size = min(final_w, final_h)

        if final_w != final_h:
            center_x = (crop_x_min + crop_x_max) // 2
            center_y = (crop_y_min + crop_y_max) // 2
            half_final = final_crop_size // 2
            crop_x_min = max(0, center_x - half_final)
            crop_y_min = max(0, center_y - half_final)
            crop_x_max = min(img_w, crop_x_min + final_crop_size)
            crop_y_max = min(img_h, crop_y_min + final_crop_size)

    return (int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max))


def _load_rgb_image(image_path: str) -> np.ndarray:
    """Load an image as RGB uint8 array.

    Prefer PIL over ``cv2.imread`` for large industrial PNGs — OpenCV's PNG
    decoder warns (and can fail) when IDAT chunks exceed its internal limit.
    """
    with Image.open(image_path) as im:
        return np.array(im.convert("RGB"))


def extract_patches_from_image_json(
    image_path: str,
    json_path: str,
    output_dir: str,
    min_size: int = 8,
    crop_sizes: List[int] = [96, 160, 224],
    label: Optional[str] = None,
) -> List[dict]:
    """从图像和 JSON 文件提取 patches。

    每个独立的非 subtract 标注（label）单独生成一张 crop 图，确保每张 crop
    只对应一个标注实例。返回 patch 信息列表（含 patch_path / patch_mask_path /
    label_index / bbox / crop_bbox / original_size / patch_size 等）。
    """
    image = _load_rgb_image(str(image_path))

    img_h, img_w = image.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    labels_data = json_data.get("labels", [])
    if len(labels_data) == 0:
        print(f"警告: {json_path} 中没有找到 labels")
        return []

    subtract_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for ld in labels_data:
        if not ld.get("isSubtract", False):
            continue
        pts = ld.get("points", [])
        if len(pts) < 3:
            continue
        subtract_mask = np.where(polygon_to_mask(pts, img_h, img_w) > 0, 255, subtract_mask)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(image_path).stem
    patches_info = []

    for label_idx, label_data in enumerate(labels_data):
        if label_data.get("isSubtract", False):
            continue

        points = label_data.get("points", [])
        if len(points) < 3:
            continue

        try:
            label_mask = polygon_to_mask(points, img_h, img_w)
            label_mask = np.where(subtract_mask > 0, 0, label_mask).astype(np.uint8)
        except Exception as exc:
            print(f"  [crop] label {label_idx}: SKIP (mask generation error: {exc})")
            continue

        bbox = extract_bbox_from_mask(label_mask)
        if bbox is None:
            continue

        x_min, y_min, x_max, y_max = bbox
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        bbox_max = max(bbox_w, bbox_h)
        if bbox_max < min_size:
            continue

        try:
            crop_x_min, crop_y_min, crop_x_max, crop_y_max = compute_crop_region(
                bbox, (img_h, img_w), crop_sizes
            )

            patch = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            patch_mask = label_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

            if patch.shape[0] != patch.shape[1]:
                max_dim = max(patch.shape[0], patch.shape[1])
                patch = cv2.resize(patch, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)
                patch_mask = cv2.resize(patch_mask, (max_dim, max_dim), interpolation=cv2.INTER_NEAREST)

            patch_filename = f"{prefix}_crop_{label_idx:03d}.png"
            patch_mask_filename = f"{prefix}_crop_{label_idx:03d}_mask.png"
            patch_path = output_dir / patch_filename
            patch_mask_path = output_dir / patch_mask_filename

            Image.fromarray(patch).save(patch_path)
            Image.fromarray(patch_mask).save(patch_mask_path)
        except Exception as exc:
            print(f"  [crop] label {label_idx}: SKIP (crop/save error: {exc})")
            continue

        patches_info.append({
            "patch_path": str(patch_path),
            "patch_mask_path": str(patch_mask_path),
            "original_image_path": str(image_path),
            "original_json_path": str(json_path),
            "instance_id": label_idx,
            "label_index": label_idx,
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "crop_bbox": [int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)],
            "original_size": [img_h, img_w],
            "patch_size": [patch.shape[0], patch.shape[1]],
        })

    return patches_info
