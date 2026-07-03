"""Background dataset import jobs with DB-backed progress."""

from __future__ import annotations

import zipfile
from io import BytesIO

from PIL import Image as PILImage

from data_cluster.app.config import Settings, get_settings
from data_cluster.app.database import SessionLocal
from data_cluster.app.models.db import Annotation, AnnotationRegion, Dataset, DefectClass, Image
from data_cluster.app.services.dataset_import_log import log_import
from data_cluster.app.services.dataset_import_progress import clear_import_progress, update_import_progress
from data_cluster.app.services.storage import (
    delete_dataset_files,
    extract_zip_to_dataset,
    human_size,
    save_upload_bytes_sync,
    url_to_fs_path,
)
from data_cluster.dl.crop import generate_crops_for_image

_import_executor = None


def _get_executor():
    global _import_executor
    if _import_executor is None:
        from concurrent.futures import ThreadPoolExecutor

        _import_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dataset-import")
    return _import_executor


def schedule_dataset_import(fn, *args) -> None:
    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_get_executor(), fn, *args)


def _image_dims(contents: bytes) -> tuple[int | None, int | None]:
    try:
        with PILImage.open(BytesIO(contents)) as im:
            return im.width, im.height
    except Exception:
        return None, None


def _get_or_create_class(db, dataset_id: str, name: str, cache: dict[str, DefectClass]) -> DefectClass:
    key = name.strip()
    if key in cache:
        return cache[key]
    existing = (
        db.query(DefectClass)
        .filter(DefectClass.dataset_id == dataset_id, DefectClass.name == key)
        .first()
    )
    if existing:
        cache[key] = existing
        return existing
    order = db.query(DefectClass).filter(DefectClass.dataset_id == dataset_id).count()
    cls = DefectClass(name=key, dataset_id=dataset_id, sort_order=order)
    db.add(cls)
    db.flush()
    cache[key] = cls
    return cls


def _fail_import(dataset_id: str, message: str) -> None:
    log_import(f"[{dataset_id}] 导入失败: {message}")
    update_import_progress(dataset_id, "failed", message=message, force=True)
    clear_import_progress(dataset_id)


def _rollback_dataset(dataset_id: str) -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        delete_dataset_files(settings, dataset_id)
        ds = db.get(Dataset, dataset_id)
        if ds:
            db.delete(ds)
            db.commit()
    finally:
        db.close()
    clear_import_progress(dataset_id)


def run_zip_import_job(dataset_id: str, name: str, raw: bytes) -> None:
    settings = get_settings()
    try:
        update_import_progress(
            dataset_id,
            "extracting",
            sub_progress=0.0,
            message="开始解压 zip…",
            force=True,
        )

        def _on_extract(current: int, total: int, detail: str) -> None:
            sub = current / total if total else 0.0
            update_import_progress(
                dataset_id,
                "extracting",
                sub_progress=sub,
                message=f"解压 ({current}/{total}) {detail}",
            )

        records, total_bytes, url_annotations = extract_zip_to_dataset(
            settings,
            dataset_id,
            raw,
            on_extract_progress=_on_extract,
        )
        _persist_zip_records(settings, dataset_id, records, total_bytes, url_annotations)
        update_import_progress(dataset_id, "done", message="导入完成", force=True)
        log_import(f"[{dataset_id}] 数据集「{name}」导入完成")
    except zipfile.BadZipFile:
        _fail_import(dataset_id, "无效的 zip 文件")
        _rollback_dataset(dataset_id)
    except Exception as exc:  # noqa: BLE001
        _fail_import(dataset_id, str(exc))
        _rollback_dataset(dataset_id)
    finally:
        clear_import_progress(dataset_id)


def run_images_import_job(
    dataset_id: str,
    name: str,
    file_payloads: list[tuple[str, bytes]],
    labels: list[str],
) -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds:
            return
        total = len(file_payloads)
        total_bytes = 0
        images: list[Image] = []
        for i, (filename, contents) in enumerate(file_payloads, 1):
            update_import_progress(
                dataset_id,
                "saving",
                sub_progress=(i - 1) / total if total else 0.0,
                message=f"保存图片 ({i}/{total})",
            )
            log_import(f"[{dataset_id}] 保存图片 ({i}/{total})")
            url, nbytes = save_upload_bytes_sync(settings, dataset_id, filename, contents)
            w, h = _image_dims(contents)
            images.append(
                Image(
                    url=url,
                    dataset_id=dataset_id,
                    file_path=str(url_to_fs_path(settings, url)),
                    width=w,
                    height=h,
                    source="batch",
                    annotation_status="unannotated",
                )
            )
            total_bytes += nbytes
        db.add_all(images)
        cache: dict[str, DefectClass] = {}
        for label in labels:
            if label.strip():
                _get_or_create_class(db, dataset_id, label, cache)
        ds.items = len(images)
        ds.size = human_size(total_bytes) if total_bytes else None
        db.commit()
        update_import_progress(dataset_id, "done", message="导入完成", force=True)
        log_import(f"[{dataset_id}] 数据集「{name}」导入完成 ({total} 张图片)")
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        _fail_import(dataset_id, str(exc))
        _rollback_dataset(dataset_id)
    finally:
        db.close()
        clear_import_progress(dataset_id)


def _persist_zip_records(
    settings: Settings,
    dataset_id: str,
    records: list[tuple[str, str]],
    total_bytes: int,
    url_annotations: dict[str, dict],
) -> None:
    db = SessionLocal()
    try:
        dataset = db.get(Dataset, dataset_id)
        if not dataset:
            return

        cache: dict[str, DefectClass] = {}
        image_count = 0
        annotated_images: list[Image] = []
        total_records = len(records)

        update_import_progress(
            dataset_id,
            "persisting",
            sub_progress=0.0,
            message=f"写入数据库: {total_records} 张图片",
            force=True,
        )
        log_import(
            f"[{dataset_id}] 写入数据库: {total_records} 张图片, "
            f"{len(url_annotations)} 张带标注"
        )

        for idx, (category_name, url) in enumerate(records, 1):
            cls = _get_or_create_class(db, dataset_id, category_name, cache)
            ann_data = url_annotations.get(url)
            annotated = bool(ann_data and ann_data.get("labels"))
            img = Image(
                url=url,
                dataset_id=dataset_id,
                file_path=str(url_to_fs_path(settings, url)),
                source="zip",
                annotation_status="annotated" if annotated else "unannotated",
            )
            db.add(img)
            db.flush()
            image_count += 1

            if ann_data:
                db.add(
                    Annotation(
                        image_id=img.id,
                        shape_type=ann_data.get("shape_type"),
                        network_type=ann_data.get("network_type"),
                        payload=dict(ann_data),
                    )
                )
                for order, label in enumerate(ann_data.get("labels") or []):
                    if not isinstance(label, dict):
                        continue
                    db.add(
                        AnnotationRegion(
                            image_id=img.id,
                            class_id=None if label.get("isSubtract") else cls.id,
                            points=label.get("points") or [],
                            is_subtract=bool(label.get("isSubtract", False)),
                            order=order,
                        )
                    )
            if annotated:
                annotated_images.append(img)

            if total_records:
                update_import_progress(
                    dataset_id,
                    "persisting",
                    sub_progress=idx / total_records,
                    message=f"写入数据库 ({idx}/{total_records})",
                )

        dataset.items = image_count
        dataset.size = human_size(total_bytes) if total_bytes else None
        db.flush()

        crop_total = len(annotated_images)
        if crop_total:
            update_import_progress(
                dataset_id,
                "cropping",
                sub_progress=0.0,
                message=f"开始生成裁剪图: {crop_total} 张原图",
                force=True,
            )
            log_import(f"[{dataset_id}] 开始生成裁剪图: {crop_total} 张原图")
        for i, img in enumerate(annotated_images, 1):
            db.expire(img, ["regions"])
            try:
                log_import(f"[{dataset_id}] 裁剪 ({i}/{crop_total}) image={img.id}")
                crops = generate_crops_for_image(
                    db=db, settings=settings, image=img, dataset_id=dataset_id
                )
                log_import(f"[{dataset_id}] 裁剪 ({i}/{crop_total}) -> {len(crops)} 个 crop")
            except Exception as exc:  # noqa: BLE001
                log_import(f"[{dataset_id}] 裁剪失败 image={img.id}: {exc}")
            if crop_total:
                update_import_progress(
                    dataset_id,
                    "cropping",
                    sub_progress=i / crop_total,
                    message=f"裁剪 ({i}/{crop_total})",
                )
        if crop_total:
            log_import(f"[{dataset_id}] 裁剪图生成完成")
        db.commit()
    finally:
        db.close()
