"""Crop service: given an Image with annotation regions, run the patch extractor
and persist the resulting CropImage rows (one per non-subtract region).

Each crop inherits the defect class of the region that produced it, so a single
image can yield crops belonging to several different defect classes.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from data_cluster.app.config import Settings
from data_cluster.app.models.db import AnnotationRegion, CropImage, Image
from data_cluster.app.services.storage import crop_images_dir, crop_static_url, url_to_fs_path
from embedding_model.preprocess.polygon_patch_extractor import extract_patches_from_image_json


def _remap_label_to_crop(
    points: list[list[float]],
    is_subtract: bool,
    crop_bbox: list[int],
    original_size: list[int],
) -> dict[str, Any] | None:
    """Re-map a region polygon from original-image space to crop-image space (0-1)."""
    img_h, img_w = original_size
    cx0, cy0, cx1, cy1 = crop_bbox
    cw = cx1 - cx0
    ch = cy1 - cy0
    if cw <= 0 or ch <= 0:
        return None

    new_pts: list[list[float]] = []
    for pt in points:
        px_x = pt[0] * img_w
        px_y = pt[1] * img_h
        nx = max(0.0, min(1.0, (px_x - cx0) / cw))
        ny = max(0.0, min(1.0, (px_y - cy0) / ch))
        new_pts.append([round(nx, 6), round(ny, 6)])
    if len(new_pts) < 3:
        return None
    return {"points": new_pts, "isSubtract": is_subtract}


def generate_crops_for_image(
    db: Session,
    settings: Settings,
    image: Image,
    dataset_id: str,
    min_size: int = 8,
    crop_sizes: list[int] | None = None,
) -> list[CropImage]:
    """Run the crop pipeline for one Image and persist results.

    All crop patches are written to ``data/.../{dataset_id}/crop_images/`` (flat).
    Existing crops for this image are replaced.  Returns the new CropImage rows.
    """
    if crop_sizes is None:
        crop_sizes = [96, 160, 224]

    # Ordered regions; index in this list is the ``label_index`` the extractor returns.
    regions: list[AnnotationRegion] = sorted(image.regions, key=lambda r: (r.order, r.created_at))
    if not regions:
        return []

    image_fs = url_to_fs_path(settings, image.url) or Path(image.file_path or "")
    if not image_fs or not image_fs.exists():
        raise FileNotFoundError(f"Image file not found on disk: {image_fs}")

    labels_payload = [
        {"points": r.points or [], "isSubtract": bool(r.is_subtract)} for r in regions
    ]
    payload = {
        "image_name": image_fs.name,
        "image_uuid": str(uuid.uuid4()),
        "image_path": str(image_fs.parent),
        "labels": labels_payload,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp)
        tmp_json_path = tmp.name

    output_dir = crop_images_dir(settings, dataset_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        patches_info = extract_patches_from_image_json(
            image_path=str(image_fs),
            json_path=tmp_json_path,
            output_dir=str(output_dir),
            min_size=min_size,
            crop_sizes=crop_sizes,
        )
    finally:
        Path(tmp_json_path).unlink(missing_ok=True)

    # Remove old crops for this image
    db.query(CropImage).filter(CropImage.source_image_id == image.id).delete()
    db.flush()

    new_crops: list[CropImage] = []
    for info in patches_info:
        patch_path = Path(info["patch_path"])
        mask_path = Path(info["patch_mask_path"])

        patch_url = crop_static_url(dataset_id, patch_path.name)
        mask_url: str | None = (
            crop_static_url(dataset_id, mask_path.name) if mask_path.exists() else None
        )

        label_index: int = info.get("label_index", info["instance_id"])
        region = regions[label_index] if 0 <= label_index < len(regions) else None

        crop_annotation: list[dict[str, Any]] | None = None
        if region is not None and region.points:
            remapped = _remap_label_to_crop(
                region.points, bool(region.is_subtract), info["crop_bbox"], info["original_size"]
            )
            if remapped is not None:
                crop_annotation = [remapped]

        crop = CropImage(
            url=patch_url,
            file_path=str(patch_path.resolve()),
            dataset_id=dataset_id,
            class_id=region.class_id if region is not None else None,
            region_id=region.id if region is not None else None,
            mask_url=mask_url,
            mask_path=str(mask_path.resolve()) if mask_path.exists() else None,
            source_image_id=image.id,
            instance_index=label_index,
            bbox=info["bbox"],
            crop_bbox=info["crop_bbox"],
            original_size=info["original_size"],
            patch_size=info["patch_size"],
            crop_annotation=crop_annotation,
        )
        db.add(crop)
        new_crops.append(crop)

    db.flush()
    return new_crops
