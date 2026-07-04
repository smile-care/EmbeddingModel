import asyncio
import json
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image as PILImage
from sqlalchemy.orm import Session, selectinload

from data_cluster.app.config import get_settings
from data_cluster.app.database import get_db
from data_cluster.app.models.db import (AnnotationRegion, CropImage, Dataset, DefectClass,
                                        Image)
from data_cluster.app.schemas.dataset import (AnnotationRegionOut, AnnotationSaveRequest,
                                              CropImageOut, DatasetCreateResponse,
                                              DatasetDetail, DatasetImportStatus,
                                              DatasetSummary, DefectClassCreate,
                                              DefectClassOut, DefectClassUpdate, ImageOut,
                                              ImageUploadResponse)
from data_cluster.dl.crop import generate_crops_for_image
from data_cluster.app.services.dataset_import_job import (run_images_import_job,
                                                            run_zip_import_job,
                                                            schedule_dataset_import,
                                                            cancel_dataset_import_job)
from data_cluster.app.services.dataset_import_progress import update_import_progress
from data_cluster.app.services.storage import (dataset_upload_dir_bytes, delete_dataset_files,
                                               ensure_upload_root, get_or_create_thumbnail,
                                               human_size, save_upload_file, url_to_fs_path)
from data_cluster.app.services.dataset_import_log import log_import

router = APIRouter(prefix="/datasets", tags=["datasets"])


# ── Serialization ─────────────────────────────────────────────────────────────

def _serialize_class(c: DefectClass) -> DefectClassOut:
    return DefectClassOut(id=c.id, name=c.name, color=c.color, sort_order=c.sort_order)


def _serialize_region(r: AnnotationRegion) -> AnnotationRegionOut:
    return AnnotationRegionOut(
        id=r.id,
        class_id=r.class_id,
        points=r.points or [],
        is_subtract=bool(r.is_subtract),
        order=r.order,
    )


def _serialize_crop(c: CropImage) -> CropImageOut:
    return CropImageOut(
        id=c.id,
        url=c.url,
        class_id=c.class_id,
        mask_url=c.mask_url,
        source_image_id=c.source_image_id,
        region_id=c.region_id,
        instance_index=c.instance_index,
        bbox=c.bbox,
        crop_bbox=c.crop_bbox,
        original_size=c.original_size,
        patch_size=c.patch_size,
        crop_annotation=c.crop_annotation,
    )


def _serialize_image(i: Image) -> ImageOut:
    return ImageOut(
        id=i.id,
        url=i.url,
        width=i.width,
        height=i.height,
        source=i.source,
        annotation_status=i.annotation_status,
        regions=[_serialize_region(r) for r in sorted(i.regions, key=lambda r: (r.order, r.created_at))],
        crops=[_serialize_crop(c) for c in sorted(i.crops, key=lambda c: c.instance_index)],
    )


def _serialize_summary(ds: Dataset, image_count: int | None = None) -> DatasetSummary:
    return DatasetSummary(
        id=ds.id,
        name=ds.name,
        type=ds.type,
        size=ds.size,
        items=image_count if image_count is not None else ds.items,
        status=ds.status,
        import_progress=ds.import_progress,
        import_stage=ds.import_stage,
        import_message=ds.import_message,
        created_at=ds.created_at,
        updated_at=ds.updated_at,
        defect_classes=[_serialize_class(c) for c in sorted(ds.defect_classes, key=lambda c: (c.sort_order, c.name))],
    )


def _serialize_detail(ds: Dataset) -> DatasetDetail:
    base = _serialize_summary(ds, image_count=len(ds.images))
    return DatasetDetail(
        **base.model_dump(by_alias=False),
        images=[_serialize_image(i) for i in sorted(ds.images, key=lambda i: i.created_at)],
    )


def _load_dataset_detail(db: Session, dataset_id: str) -> Dataset | None:
    return (
        db.query(Dataset)
        .options(
            selectinload(Dataset.defect_classes),
            selectinload(Dataset.images).selectinload(Image.regions),
            selectinload(Dataset.images).selectinload(Image.crops),
        )
        .filter(Dataset.id == dataset_id)
        .first()
    )


def _get_or_create_class(db: Session, dataset_id: str, name: str, cache: dict[str, DefectClass]) -> DefectClass:
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


# ── Dataset CRUD ──────────────────────────────────────────────────────────────

@router.get("", response_model=list[DatasetSummary])
async def list_datasets(db: Session = Depends(get_db)) -> list[DatasetSummary]:
    loop = asyncio.get_event_loop()

    def _query():
        from sqlalchemy import func
        rows = (
            db.query(Dataset)
            .options(selectinload(Dataset.defect_classes))
            .order_by(Dataset.created_at.desc())
            .all()
        )
        counts = dict(
            db.query(Image.dataset_id, func.count(Image.id))
            .group_by(Image.dataset_id)
            .all()
        )
        return [(ds, counts.get(ds.id, 0)) for ds in rows]

    rows = await loop.run_in_executor(None, _query)
    return [_serialize_summary(ds, image_count=cnt) for ds, cnt in rows]


# Registered before "/{dataset_id}" so the literal path is not captured as an id.
@router.get("/thumbnail")
async def get_thumbnail(
    url: str = Query(..., description="A /static/... image URL to thumbnail"),
    size: int = Query(384, ge=32, le=1024),
) -> FileResponse:
    """Serve a small cached JPEG thumbnail for grid previews (lazy-generated)."""
    if not url.startswith("/static/"):
        raise HTTPException(status_code=400, detail="url must be a /static/ path")
    settings = get_settings()
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, get_or_create_thumbnail, settings, url, size)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/{dataset_id}/import-status", response_model=DatasetImportStatus)
async def get_import_status(dataset_id: str, db: Session = Depends(get_db)) -> DatasetImportStatus:
    loop = asyncio.get_event_loop()

    def _query():
        return db.get(Dataset, dataset_id)

    ds = await loop.run_in_executor(None, _query)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetImportStatus(
        id=ds.id,
        status=ds.status,
        items=ds.items,
        import_progress=ds.import_progress,
        import_stage=ds.import_stage,
        import_message=ds.import_message,
    )


@router.post("/{dataset_id}/cancel-import")
async def cancel_import(dataset_id: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    """Cancel an in-flight dataset import and delete partial artifacts."""
    loop = asyncio.get_event_loop()

    def _cancel():
        ds = db.get(Dataset, dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if ds.status not in {"Processing", "Failed"}:
            raise HTTPException(status_code=409, detail="Dataset is not importing")
        cancel_dataset_import_job(dataset_id)

    await loop.run_in_executor(None, _cancel)
    return {"success": True}


@router.get("/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)) -> DatasetDetail:
    loop = asyncio.get_event_loop()
    ds = await loop.run_in_executor(None, _load_dataset_detail, db, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _serialize_detail(ds)


@router.post("", response_model=DatasetCreateResponse)
async def create_dataset(
    name: str = Form(...),
    type: str = Form("Image"),
    zip_file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    category_labels: str | None = Form(None),
    db: Session = Depends(get_db),
) -> DatasetCreateResponse:
    """Create a dataset.

    Two bulk-upload paths are supported:
      * ``zip_file``  - folder-per-class archive with optional ``*.json`` sidecars
                        (original images + labels in one click).
      * ``files[]``   - plain images (no labels); stored unannotated for later
                        in-browser mask annotation.
    """
    settings = get_settings()
    ensure_upload_root(settings)

    file_list = files or []
    labels: list[str] = []
    if category_labels:
        try:
            data = json.loads(category_labels)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="category_labels must be valid JSON") from e
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="category_labels must be a JSON array")
        labels = [str(x) for x in data]

    use_zip = bool(zip_file and zip_file.filename and zip_file.filename.lower().endswith(".zip"))

    loop = asyncio.get_event_loop()

    def _create_ds():
        ds = Dataset(
            name=name,
            type=type,
            items=0,
            status="Processing",
            size=None,
            import_stage="uploading",
            import_progress=0.0,
            import_message="正在接收文件…",
        )
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return ds

    ds = await loop.run_in_executor(None, _create_ds)

    try:
        if use_zip and zip_file:
            log_import(f"[{ds.id}] 上传数据集「{name}」: 开始处理 zip ({zip_file.filename})")
            raw = await zip_file.read()
            log_import(f"[{ds.id}] zip 已接收 ({human_size(len(raw))})")
            update_import_progress(
                ds.id,
                "extracting",
                sub_progress=0.0,
                message="文件已接收，开始解压…",
                force=True,
            )
            schedule_dataset_import(run_zip_import_job, ds.id, name, raw)
        elif file_list:
            log_import(f"[{ds.id}] 上传数据集「{name}」: 开始处理 {len(file_list)} 张图片")
            contents_list = await asyncio.gather(*[f.read() for f in file_list])
            payloads = [
                (upload.filename or f"image_{i}.jpg", contents)
                for i, (upload, contents) in enumerate(zip(file_list, contents_list), 1)
            ]
            update_import_progress(
                ds.id,
                "saving",
                sub_progress=0.0,
                message=f"开始保存 {len(payloads)} 张图片…",
                force=True,
            )
            schedule_dataset_import(run_images_import_job, ds.id, name, payloads, labels)
        else:
            def _mark_ready():
                ds.status = "Ready"
                ds.import_stage = "done"
                ds.import_progress = 100.0
                ds.import_message = None
                db.commit()
                db.refresh(ds)

            await loop.run_in_executor(None, _mark_ready)

    except HTTPException:
        await loop.run_in_executor(None, _rollback_dataset, db, settings, ds.id)
        raise
    except Exception:
        await loop.run_in_executor(None, _rollback_dataset, db, settings, ds.id)
        raise

    return DatasetCreateResponse(id=ds.id, name=ds.name, items=ds.items, status=ds.status)


def _rollback_dataset(db: Session, settings, dataset_id: str) -> None:
    db.rollback()
    delete_dataset_files(settings, dataset_id)
    ds = db.get(Dataset, dataset_id)
    if ds:
        db.delete(ds)
        db.commit()


def _image_dims(contents: bytes) -> tuple[int | None, int | None]:
    try:
        with PILImage.open(BytesIO(contents)) as im:
            return im.width, im.height
    except Exception:
        return None, None


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    loop = asyncio.get_event_loop()

    def _delete():
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        settings = get_settings()
        delete_dataset_files(settings, dataset_id)
        db.delete(ds)
        db.commit()

    await loop.run_in_executor(None, _delete)
    return {"success": True}


# ── Defect class registry ─────────────────────────────────────────────────────

@router.get("/{dataset_id}/classes", response_model=list[DefectClassOut])
async def list_classes(dataset_id: str, db: Session = Depends(get_db)) -> list[DefectClassOut]:
    loop = asyncio.get_event_loop()

    def _query():
        return (
            db.query(DefectClass)
            .filter(DefectClass.dataset_id == dataset_id)
            .order_by(DefectClass.sort_order, DefectClass.name)
            .all()
        )

    rows = await loop.run_in_executor(None, _query)
    return [_serialize_class(c) for c in rows]


@router.post("/{dataset_id}/classes", response_model=DefectClassOut)
async def create_class(
    dataset_id: str, body: DefectClassCreate, db: Session = Depends(get_db)
) -> DefectClassOut:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    loop = asyncio.get_event_loop()

    def _create():
        if not db.get(Dataset, dataset_id):
            raise HTTPException(status_code=404, detail="Dataset not found")
        existing = (
            db.query(DefectClass)
            .filter(DefectClass.dataset_id == dataset_id, DefectClass.name == name)
            .first()
        )
        if existing:
            return existing
        order = db.query(DefectClass).filter(DefectClass.dataset_id == dataset_id).count()
        cls = DefectClass(name=name, color=body.color, dataset_id=dataset_id, sort_order=order)
        db.add(cls)
        db.commit()
        db.refresh(cls)
        return cls

    cls = await loop.run_in_executor(None, _create)
    return _serialize_class(cls)


@router.patch("/classes/{class_id}", response_model=DefectClassOut)
async def update_class(
    class_id: str, body: DefectClassUpdate, db: Session = Depends(get_db)
) -> DefectClassOut:
    loop = asyncio.get_event_loop()

    def _update():
        cls = db.get(DefectClass, class_id)
        if not cls:
            raise HTTPException(status_code=404, detail="Class not found")
        if body.name is not None and body.name.strip():
            cls.name = body.name.strip()
        if body.color is not None:
            cls.color = body.color
        db.commit()
        db.refresh(cls)
        return cls

    cls = await loop.run_in_executor(None, _update)
    return _serialize_class(cls)


@router.delete("/classes/{class_id}")
async def delete_class(class_id: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    loop = asyncio.get_event_loop()

    def _delete():
        cls = db.get(DefectClass, class_id)
        if not cls:
            raise HTTPException(status_code=404, detail="Class not found")
        db.delete(cls)
        db.commit()

    await loop.run_in_executor(None, _delete)
    return {"success": True}


# ── Image upload (single / batch, no labels) ──────────────────────────────────

@router.post("/{dataset_id}/images", response_model=DatasetDetail)
async def add_images_to_dataset(
    dataset_id: str,
    files: list[UploadFile] = File(...),
    source: str = Form("single"),
    db: Session = Depends(get_db),
) -> DatasetDetail:
    """Upload one or more images (no class assignment). Stored as unannotated."""
    file_list = files or []
    if not file_list:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    settings = get_settings()
    ensure_upload_root(settings)
    loop = asyncio.get_event_loop()

    ds = await loop.run_in_executor(None, lambda: db.get(Dataset, dataset_id))
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    contents_list = await asyncio.gather(*[f.read() for f in file_list])
    results = await asyncio.gather(*[
        save_upload_file(settings, dataset_id, "", upload, contents=contents)
        for upload, contents in zip(file_list, contents_list)
    ])

    def _persist():
        new_images = []
        for i, ((url, _nbytes), contents) in enumerate(zip(results, contents_list), 1):
            log_import(f"[{dataset_id}] 追加图片 ({i}/{len(file_list)})")
            w, h = _image_dims(contents)
            new_images.append(Image(
                url=url,
                dataset_id=dataset_id,
                file_path=str(url_to_fs_path(settings, url)),
                width=w,
                height=h,
                source=source if source in {"single", "batch", "zip"} else "single",
                annotation_status="unannotated",
            ))
        db.add_all(new_images)
        ds.status = "Ready"
        total_bytes = dataset_upload_dir_bytes(settings, dataset_id)
        ds.size = human_size(total_bytes) if total_bytes else None
        ds.items = db.query(Image).filter(Image.dataset_id == dataset_id).count()
        db.commit()

    await loop.run_in_executor(None, _persist)
    log_import(f"[{dataset_id}] 追加完成 ({len(file_list)} 张图片)")
    out = await loop.run_in_executor(None, _load_dataset_detail, db, dataset_id)
    assert out is not None
    return _serialize_detail(out)


@router.post("/{dataset_id}/images/single", response_model=ImageUploadResponse)
async def upload_single_image(
    dataset_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ImageUploadResponse:
    """Upload exactly one image and return its record (for the annotation flow)."""
    settings = get_settings()
    ensure_upload_root(settings)
    loop = asyncio.get_event_loop()

    ds = await loop.run_in_executor(None, lambda: db.get(Dataset, dataset_id))
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    contents = await file.read()
    url, _nbytes = await save_upload_file(settings, dataset_id, "", file, contents=contents)
    w, h = _image_dims(contents)

    def _persist():
        img = Image(
            url=url,
            dataset_id=dataset_id,
            file_path=str(url_to_fs_path(settings, url)),
            width=w,
            height=h,
            source="single",
            annotation_status="unannotated",
        )
        db.add(img)
        ds.status = "Ready"
        total_bytes = dataset_upload_dir_bytes(settings, dataset_id)
        ds.size = human_size(total_bytes) if total_bytes else None
        ds.items = db.query(Image).filter(Image.dataset_id == dataset_id).count()
        db.commit()
        db.refresh(img)
        return img

    img = await loop.run_in_executor(None, _persist)
    return ImageUploadResponse(image=_serialize_image(img))


@router.delete("/images/{image_id}")
async def delete_image(image_id: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    loop = asyncio.get_event_loop()

    def _delete():
        img = db.get(Image, image_id)
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        ds = db.get(Dataset, img.dataset_id)
        db.delete(img)
        if ds:
            db.flush()
            ds.items = db.query(Image).filter(Image.dataset_id == ds.id).count()
        db.commit()

    await loop.run_in_executor(None, _delete)
    return {"success": True}


# ── Annotation ────────────────────────────────────────────────────────────────

@router.get("/images/{image_id}/annotation", response_model=list[AnnotationRegionOut])
async def get_annotation(image_id: str, db: Session = Depends(get_db)) -> list[AnnotationRegionOut]:
    loop = asyncio.get_event_loop()

    def _query():
        return (
            db.query(AnnotationRegion)
            .filter(AnnotationRegion.image_id == image_id)
            .order_by(AnnotationRegion.order)
            .all()
        )

    rows = await loop.run_in_executor(None, _query)
    return [_serialize_region(r) for r in rows]


@router.put("/images/{image_id}/annotation", response_model=ImageOut)
async def save_annotation(
    image_id: str, body: AnnotationSaveRequest, db: Session = Depends(get_db)
) -> ImageOut:
    """Replace all regions for an image, then optionally (re)generate crops."""
    settings = get_settings()
    loop = asyncio.get_event_loop()

    def _save():
        img = (
            db.query(Image)
            .options(selectinload(Image.regions))
            .filter(Image.id == image_id)
            .first()
        )
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")

        db.query(AnnotationRegion).filter(AnnotationRegion.image_id == image_id).delete()
        db.flush()

        has_region = False
        for order, region in enumerate(body.regions):
            pts = region.points or []
            if len(pts) < 3:
                continue
            db.add(AnnotationRegion(
                image_id=image_id,
                class_id=None if region.is_subtract else region.class_id,
                points=pts,
                is_subtract=bool(region.is_subtract),
                order=order,
            ))
            has_region = True

        img.annotation_status = "annotated" if has_region else "unannotated"
        db.flush()
        db.expire(img, ["regions", "crops"])

        if body.generate_crops:
            db.query(CropImage).filter(CropImage.source_image_id == image_id).delete()
            db.flush()
            if has_region:
                try:
                    generate_crops_for_image(
                        db=db, settings=settings, image=img, dataset_id=img.dataset_id
                    )
                except FileNotFoundError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                except Exception as exc:  # noqa: BLE001
                    db.rollback()
                    raise HTTPException(status_code=500, detail=f"Crop failed: {exc}") from exc

        db.commit()
        out = (
            db.query(Image)
            .options(selectinload(Image.regions), selectinload(Image.crops))
            .filter(Image.id == image_id)
            .first()
        )
        return out

    img = await loop.run_in_executor(None, _save)
    assert img is not None
    return _serialize_image(img)


# ── Crops ─────────────────────────────────────────────────────────────────────

@router.get("/classes/{class_id}/crops", response_model=list[CropImageOut])
async def get_crops_for_class(class_id: str, db: Session = Depends(get_db)) -> list[CropImageOut]:
    """Return all crop patches belonging to a defect class."""
    loop = asyncio.get_event_loop()

    def _query():
        return (
            db.query(CropImage)
            .filter(CropImage.class_id == class_id)
            .order_by(CropImage.source_image_id, CropImage.instance_index)
            .all()
        )

    crops = await loop.run_in_executor(None, _query)
    return [_serialize_crop(c) for c in crops]


@router.get("/images/{image_id}/crops", response_model=list[CropImageOut])
async def get_crops(image_id: str, db: Session = Depends(get_db)) -> list[CropImageOut]:
    loop = asyncio.get_event_loop()
    crops = await loop.run_in_executor(
        None,
        lambda: db.query(CropImage).filter(CropImage.source_image_id == image_id).all()
    )
    return [_serialize_crop(c) for c in crops]


@router.post("/images/{image_id}/crops", response_model=list[CropImageOut])
async def create_crops(
    image_id: str,
    min_size: int = Query(default=8, ge=1, le=500, description="Minimum defect bbox size in pixels to generate a crop"),
    db: Session = Depends(get_db),
) -> list[CropImageOut]:
    """Trigger crop generation for one image.  Idempotent: replaces existing crops."""
    loop = asyncio.get_event_loop()

    def _load_image():
        return (
            db.query(Image)
            .options(selectinload(Image.regions))
            .filter(Image.id == image_id)
            .first()
        )

    image = await loop.run_in_executor(None, _load_image)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    if not image.regions:
        raise HTTPException(
            status_code=422, detail="Image has no annotation regions; cannot generate crops."
        )

    settings = get_settings()

    def _generate():
        try:
            crops = generate_crops_for_image(
                db=db,
                settings=settings,
                image=image,
                dataset_id=image.dataset_id,
                min_size=min_size,
            )
            db.commit()
            for c in crops:
                db.refresh(c)
            return crops
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Crop failed: {exc}") from exc

    new_crops = await loop.run_in_executor(None, _generate)
    return [_serialize_crop(c) for c in new_crops]
