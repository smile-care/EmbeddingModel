#!/usr/bin/env python3
"""Convert MVTec AD test + ground_truth into test_cluster_data format."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np


def mask_to_labels(mask: np.ndarray, img_h: int, img_w: int) -> list[dict]:
    """Convert binary mask to normalized polygon labels."""
    binary = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    labels: list[dict] = []
    label_id = 1
    for contour in contours:
        if cv2.contourArea(contour) < 1:
            continue
        points = [
            [float(x) / img_w, float(y) / img_h]
            for x, y in contour.reshape(-1, 2)
        ]
        if len(points) < 3:
            continue
        labels.append(
            {
                "label_id": label_id,
                "points": points,
                "isSubtract": False,
            }
        )
        label_id += 1
    return labels


def convert_category(
    defect_type: str,
    test_dir: Path,
    gt_dir: Path,
    output_root: Path,
) -> tuple[int, int]:
    """Convert one defect category. Returns (image_count, skipped_count)."""
    out_dir = output_root / defect_type
    out_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    skipped = 0

    for image_path in sorted(test_dir.glob("*.png")):
        mask_path = gt_dir / f"{image_path.stem}_mask.png"
        if not mask_path.exists():
            skipped += 1
            print(f"  skip (no mask): {defect_type}/{image_path.name}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped += 1
            print(f"  skip (bad mask): {defect_type}/{mask_path.name}")
            continue

        img_h, img_w = mask.shape[:2]
        labels = mask_to_labels(mask, img_h, img_w)
        if not labels:
            skipped += 1
            print(f"  skip (empty mask): {defect_type}/{mask_path.name}")
            continue

        image_uuid = str(uuid.uuid4())
        dest_image = out_dir / f"{image_uuid}.png"
        dest_json = out_dir / f"{image_uuid}.json"

        shutil.copy2(image_path, dest_image)

        annotation = {
            "image_name": dest_image.name,
            "image_uuid": image_uuid,
            "image_path": str(image_path.resolve()),
            "network_type": "segment",
            "shape_type": "polygon",
            "labels": labels,
        }
        dest_json.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        image_count += 1

    return image_count, skipped


def convert_mvtec(
    source_root: Path,
    output_root: Path,
    exclude: set[str] | None = None,
) -> None:
    """Convert MVTec AD category from test/ + ground_truth/ to cluster format."""
    exclude = exclude or {"good"}
    test_root = source_root / "test"
    gt_root = source_root / "ground_truth"

    if not test_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError(f"Expected test/ and ground_truth/ under {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_skipped = 0

    defect_types = sorted(
        d.name
        for d in test_root.iterdir()
        if d.is_dir() and d.name not in exclude
    )

    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Categories: {defect_types}")

    for defect_type in defect_types:
        gt_dir = gt_root / defect_type
        if not gt_dir.is_dir():
            print(f"  skip category (no ground_truth): {defect_type}")
            continue

        count, skipped = convert_category(
            defect_type,
            test_root / defect_type,
            gt_dir,
            output_root,
        )
        total_images += count
        total_skipped += skipped
        print(f"  {defect_type}: {count} images, {skipped} skipped")

    print(f"Done: {total_images} images written, {total_skipped} skipped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/mvtec_anomaly_detection/capsule"),
        help="MVTec category root containing test/ and ground_truth/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/test_cluster_data/mvtec/capsule"),
        help="Output directory (628K-style layout)",
    )
    args = parser.parse_args()
    convert_mvtec(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
