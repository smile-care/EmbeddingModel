import asyncio
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, selectinload

from data_cluster.app.config import get_settings
from data_cluster.app.database import get_db
from data_cluster.app.models.db import (CropImage, Dataset, Experiment, InferenceRun,
                                        InferenceRunProjection, InferenceUploadCategory)
from data_cluster.app.schemas.inference import (AnalyzeRequest, AnalyzeResponse,
                                                InferenceRunAnalyzeUpload, InferenceRunCreate,
                                                InferenceRunDetail, InferenceRunPatch,
                                                InferenceRunSummary, ModelInfo, PlotPoint,
                                                ProjectionOut, UploadCategoryCreate,
                                                UploadCategoryOut, UploadCategoryPatch)
from data_cluster.app.services.storage import url_to_fs_path
from data_cluster.dl.projection import (anomaly_scores_per_class, anomaly_scores_vs_golden,
                                        project_2d)
from data_cluster.dl.inference import (DEFAULT_MODEL_ID, DEFAULT_MODEL_NAME,
                                       compute_default_embeddings,
                                       compute_supcon_embeddings)

router = APIRouter(tags=["inference"])

_DEFAULT_UPLOAD_CATEGORIES = ("Category A", "Category B", "Category C")


def _ensure_default_upload_categories(db: Session) -> None:
    n = db.query(InferenceUploadCategory).count()
    if n > 0:
        return
    for i, name in enumerate(_DEFAULT_UPLOAD_CATEGORIES):
        db.add(InferenceUploadCategory(name=name, sort_order=i))
    db.commit()


def _serialize_upload_category(c: InferenceUploadCategory) -> UploadCategoryOut:
    return UploadCategoryOut(id=c.id, name=c.name, sort_order=c.sort_order)


@router.get("/inference/upload-categories", response_model=list[UploadCategoryOut])
async def list_inference_upload_categories(db: Session = Depends(get_db)) -> list[UploadCategoryOut]:
    loop = asyncio.get_event_loop()

    def _query():
        _ensure_default_upload_categories(db)
        return (
            db.query(InferenceUploadCategory)
            .order_by(InferenceUploadCategory.sort_order, InferenceUploadCategory.created_at)
            .all()
        )

    rows = await loop.run_in_executor(None, _query)
    return [_serialize_upload_category(c) for c in rows]


@router.post("/inference/upload-categories", response_model=UploadCategoryOut)
async def create_inference_upload_category(
    body: UploadCategoryCreate,
    db: Session = Depends(get_db),
) -> UploadCategoryOut:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    loop = asyncio.get_event_loop()

    def _create():
        _ensure_default_upload_categories(db)
        max_order = db.query(InferenceUploadCategory).count()
        row = InferenceUploadCategory(name=name, sort_order=max_order)
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    row = await loop.run_in_executor(None, _create)
    return _serialize_upload_category(row)


@router.patch("/inference/upload-categories/{category_id}", response_model=UploadCategoryOut)
async def patch_inference_upload_category(
    category_id: str,
    body: UploadCategoryPatch,
    db: Session = Depends(get_db),
) -> UploadCategoryOut:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    loop = asyncio.get_event_loop()

    def _patch():
        row = db.get(InferenceUploadCategory, category_id)
        if not row:
            raise HTTPException(status_code=404, detail="Category not found")
        row.name = name
        db.commit()
        db.refresh(row)
        return row

    row = await loop.run_in_executor(None, _patch)
    return _serialize_upload_category(row)


@router.delete("/inference/upload-categories/{category_id}")
async def delete_inference_upload_category(category_id: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    loop = asyncio.get_event_loop()

    def _delete():
        row = db.get(InferenceUploadCategory, category_id)
        if not row:
            raise HTTPException(status_code=404, detail="Category not found")
        db.delete(row)
        db.commit()

    await loop.run_in_executor(None, _delete)
    return {"success": True}


def _run_to_summary(run: InferenceRun, db: Session) -> InferenceRunSummary:
    ds_name = None
    if run.dataset_id:
        ds = db.get(Dataset, run.dataset_id)
        if ds:
            ds_name = ds.name
    return InferenceRunSummary(
        id=run.id,
        name=run.name,
        status=run.status,
        model_id=run.model_id,
        dataset_mode=run.dataset_mode,
        dataset_id=run.dataset_id,
        dataset_name=ds_name,
        golden_crop_ids=list(run.golden_crop_ids or []),
        algorithm=run.algorithm,
        view_mode=run.view_mode,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _run_to_detail(run: InferenceRun, db: Session) -> InferenceRunDetail:
    s = _run_to_summary(run, db)
    base = s.model_dump(by_alias=False, mode="python")
    cached = (
        db.query(InferenceRunProjection.algorithm)
        .filter(InferenceRunProjection.run_id == run.id)
        .all()
    )
    cached_algos = [r[0] for r in cached]
    return InferenceRunDetail(**base, result_json=run.result_json, cached_algorithms=cached_algos)


def _crop_fs_path(settings, crop: CropImage) -> Path:
    if crop.file_path:
        file_path = Path(crop.file_path)
        if file_path.is_file():
            return file_path
    p = url_to_fs_path(settings, crop.url)
    return p if p and p.is_file() else settings.upload_dir / "_missing_"


def _crop_mask_fs_path(settings, crop: CropImage) -> Path | None:
    if crop.mask_path:
        mask_path = Path(crop.mask_path)
        if mask_path.is_file():
            return mask_path
    if crop.mask_url:
        p = url_to_fs_path(settings, crop.mask_url)
        if p and p.is_file():
            return p
    return None


def _dataset_crop_rows(ds: Dataset) -> list[tuple[CropImage, str, int]]:
    crop_rows = [
        crop for crop in ds.crop_images
        if crop.defect_class is not None and crop.url
    ]
    label_names = sorted({crop.defect_class.name for crop in crop_rows if crop.defect_class is not None})
    name_to_idx = {name: idx for idx, name in enumerate(label_names)}

    rows: list[tuple[CropImage, str, int]] = []
    for crop in sorted(
        crop_rows,
        key=lambda c: (
            c.defect_class.name if c.defect_class is not None else "",
            c.source_image_id,
            c.instance_index,
            c.id,
        ),
    ):
        assert crop.defect_class is not None
        label_name = crop.defect_class.name
        rows.append((crop, label_name, name_to_idx[label_name]))
    return rows


@router.get("/inference/runs", response_model=list[InferenceRunSummary])
async def list_inference_runs(db: Session = Depends(get_db)) -> list[InferenceRunSummary]:
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None,
        lambda: db.query(InferenceRun).order_by(InferenceRun.created_at.desc()).all()
    )
    return [_run_to_summary(r, db) for r in rows]


@router.post("/inference/runs", response_model=InferenceRunDetail)
async def create_inference_run(body: InferenceRunCreate, db: Session = Depends(get_db)) -> InferenceRunDetail:
    loop = asyncio.get_event_loop()

    def _create():
        name = body.name.strip() or "Untitled inference"
        row = InferenceRun(
            name=name,
            status="Draft",
            model_id=body.model_id,
            dataset_mode=body.dataset_mode,
            dataset_id=body.dataset_id,
            golden_crop_ids=list(body.golden_crop_ids or []) or None,
            algorithm=body.algorithm.lower() if body.algorithm else "tsne",
            view_mode=body.view_mode,
            result_json=None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    row = await loop.run_in_executor(None, _create)
    return _run_to_detail(row, db)


@router.get("/inference/runs/{run_id}", response_model=InferenceRunDetail)
async def get_inference_run(run_id: str, db: Session = Depends(get_db)) -> InferenceRunDetail:
    loop = asyncio.get_event_loop()
    row = await loop.run_in_executor(None, lambda: db.get(InferenceRun, run_id))
    if not row:
        raise HTTPException(status_code=404, detail="Inference run not found")
    return _run_to_detail(row, db)


@router.patch("/inference/runs/{run_id}", response_model=InferenceRunDetail)
async def patch_inference_run(
    run_id: str,
    body: InferenceRunPatch,
    db: Session = Depends(get_db),
) -> InferenceRunDetail:
    loop = asyncio.get_event_loop()

    def _patch():
        row = db.get(InferenceRun, run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Inference run not found")
        data = body.model_dump(exclude_unset=True, by_alias=True)
        if "name" in data and data["name"] is not None:
            row.name = data["name"].strip() or row.name
        if "modelId" in data:
            row.model_id = data["modelId"]
        if "datasetMode" in data:
            row.dataset_mode = data["datasetMode"]
        if "datasetId" in data:
            row.dataset_id = data["datasetId"]
        if "goldenCropIds" in data:
            row.golden_crop_ids = list(data["goldenCropIds"] or []) or None
        if "algorithm" in data and data["algorithm"] is not None:
            row.algorithm = str(data["algorithm"]).lower()
        if "viewMode" in data and data["viewMode"] is not None:
            row.view_mode = data["viewMode"]
        db.commit()
        db.refresh(row)
        return row

    row = await loop.run_in_executor(None, _patch)
    return _run_to_detail(row, db)


@router.delete("/inference/runs/{run_id}")
async def delete_inference_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    loop = asyncio.get_event_loop()

    def _delete():
        row = db.get(InferenceRun, run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Inference run not found")
        db.delete(row)
        db.commit()

    await loop.run_in_executor(None, _delete)
    return {"success": True}


def _upsert_projection(db: Session, run_id: str, algorithm: str, labels: list, points: list) -> None:
    existing = (
        db.query(InferenceRunProjection)
        .filter(InferenceRunProjection.run_id == run_id, InferenceRunProjection.algorithm == algorithm)
        .first()
    )
    if existing:
        existing.labels = labels
        existing.points = points
    else:
        db.add(InferenceRunProjection(run_id=run_id, algorithm=algorithm, labels=labels, points=points))


@router.post("/inference/runs/{run_id}/analyze", response_model=InferenceRunDetail)
async def execute_inference_run(
    run_id: str,
    db: Session = Depends(get_db),
    body: InferenceRunAnalyzeUpload | None = None,
) -> InferenceRunDetail:
    loop = asyncio.get_event_loop()
    row = await loop.run_in_executor(None, lambda: db.get(InferenceRun, run_id))
    if not row:
        raise HTTPException(status_code=404, detail="Inference run not found")

    try:
        if row.dataset_mode == "existing":
            if not row.dataset_id:
                raise HTTPException(status_code=400, detail="datasetId is required for existing dataset mode")
            req = AnalyzeRequest(
                dataset_id=row.dataset_id,
                method=row.algorithm.lower(),
                experiment_id=row.model_id,
                model_id=row.model_id,
                golden_crop_ids=list(row.golden_crop_ids or []),
            )
            # 推理计算（GPU/CPU 密集）放线程池，不阻塞事件循环
            resp = await loop.run_in_executor(None, _analyze_embeddings, req, db)
            labels = list(resp.labels)
            points = [p.model_dump(by_alias=True) for p in resp.points]

            def _save_result():
                row.result_json = {"labels": labels, "points": points}
                row.status = "Completed"
                _upsert_projection(db, run_id, row.algorithm.lower(), labels, points)
                db.commit()
                db.refresh(row)

            await loop.run_in_executor(None, _save_result)
        else:
            if body is None:
                raise HTTPException(
                    status_code=400,
                    detail="Upload mode requires a JSON body with labels and points",
                )

            def _save_upload():
                labels = list(body.labels)
                points = body.points
                row.result_json = {"labels": labels, "points": points}
                row.status = "Completed"
                _upsert_projection(db, run_id, row.algorithm.lower(), labels, points)
                db.commit()
                db.refresh(row)

            await loop.run_in_executor(None, _save_upload)

        return _run_to_detail(row, db)
    except HTTPException:
        raise
    except Exception:
        def _mark_failed():
            row.status = "Failed"
            db.commit()
        await loop.run_in_executor(None, _mark_failed)
        raise


@router.get("/inference/runs/{run_id}/projections/{algorithm}", response_model=ProjectionOut)
async def get_run_projection(
    run_id: str,
    algorithm: str,
    db: Session = Depends(get_db),
) -> ProjectionOut:
    algo = algorithm.lower()
    loop = asyncio.get_event_loop()

    def _query():
        return (
            db.query(InferenceRunProjection)
            .filter(InferenceRunProjection.run_id == run_id, InferenceRunProjection.algorithm == algo)
            .first()
        )

    proj = await loop.run_in_executor(None, _query)
    if not proj:
        raise HTTPException(status_code=404, detail=f"No cached projection for algorithm '{algo}'")
    return ProjectionOut(algorithm=proj.algorithm, labels=proj.labels, points=proj.points)


@router.post("/inference/runs/{run_id}/projections/{algorithm}", response_model=ProjectionOut)
async def compute_run_projection(
    run_id: str,
    algorithm: str,
    db: Session = Depends(get_db),
) -> ProjectionOut:
    algo = algorithm.lower()
    if algo not in ("tsne", "umap", "pca"):
        raise HTTPException(status_code=400, detail="algorithm must be one of: tsne, umap, pca")

    loop = asyncio.get_event_loop()
    row = await loop.run_in_executor(None, lambda: db.get(InferenceRun, run_id))
    if not row:
        raise HTTPException(status_code=404, detail="Inference run not found")
    if row.status != "Completed":
        raise HTTPException(status_code=400, detail="Run must be in Completed status before switching projections")
    if row.dataset_mode != "existing":
        raise HTTPException(status_code=400, detail="Projection switching is only supported for existing dataset mode")
    if not row.dataset_id:
        raise HTTPException(status_code=400, detail="datasetId is required")

    req = AnalyzeRequest(
        dataset_id=row.dataset_id,
        method=algo,
        experiment_id=row.model_id,
        model_id=row.model_id,
        golden_crop_ids=list(row.golden_crop_ids or []),
    )
    resp = await loop.run_in_executor(None, _analyze_embeddings, req, db)
    labels = list(resp.labels)
    points = [p.model_dump(by_alias=True) for p in resp.points]

    def _save():
        _upsert_projection(db, run_id, algo, labels, points)
        if row.algorithm.lower() == algo:
            row.result_json = {"labels": labels, "points": points}
        db.commit()

    await loop.run_in_executor(None, _save)
    return ProjectionOut(algorithm=algo, labels=labels, points=points)


@router.get("/models", response_model=list[ModelInfo])
async def list_models(db: Session = Depends(get_db)) -> list[ModelInfo]:
    """Return the default pretrained model first, then any completed experiments."""
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None,
        lambda: db.query(Experiment)
        .filter(Experiment.status == "Completed")
        .order_by(Experiment.created_at.desc())
        .all()
    )
    models = [ModelInfo(id=DEFAULT_MODEL_ID, name=DEFAULT_MODEL_NAME, type="Pretrained")]
    models.extend(ModelInfo(id=e.id, name=e.name, type="Trained") for e in rows)
    return models


def _analyze_embeddings(req: AnalyzeRequest, db: Session) -> AnalyzeResponse:
    settings = get_settings()
    ds = (
        db.query(Dataset)
        .options(
            selectinload(Dataset.crop_images).selectinload(CropImage.defect_class),
        )
        .filter(Dataset.id == req.dataset_id)
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    eid = req.experiment_id or req.model_id
    use_default = (not eid) or eid == DEFAULT_MODEL_ID

    exp: Experiment | None = None
    checkpoint_path: Path | None = None
    if not use_default:
        exp = db.get(Experiment, eid)
        if not exp:
            # Unknown model id: fall back to the default pretrained backbone.
            use_default = True
        else:
            checkpoint = exp.checkpoint_path
            if not checkpoint:
                raise HTTPException(status_code=422, detail="所选实验没有 checkpoint_path，无法进行推理。")
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_file():
                raise HTTPException(status_code=422, detail=f"模型 checkpoint 不存在: {checkpoint_path}")

    rows = _dataset_crop_rows(ds)
    label_names = sorted({label_name for _crop, label_name, _li in rows})

    if not rows:
        raise HTTPException(
            status_code=422,
            detail="当前数据集没有可用于分析的 crop 图片，请先生成裁剪图。",
        )

    paths = []
    mask_paths = []
    label_indices = []
    for crop, _name, li in rows:
        paths.append(_crop_fs_path(settings, crop))
        mask_paths.append(_crop_mask_fs_path(settings, crop))
        label_indices.append(li)
    label_arr = np.array(label_indices, dtype=int)

    if use_default:
        emb = compute_default_embeddings(image_paths=paths, mask_paths=mask_paths)
    else:
        assert checkpoint_path is not None and exp is not None
        emb = compute_supcon_embeddings(
            checkpoint_path=checkpoint_path,
            image_paths=paths,
            mask_paths=mask_paths,
            exp_config=exp.config if isinstance(exp.config, dict) else None,
        )
    coords = project_2d(emb, req.method)

    # Golden-anchored anomaly when reference crops are supplied; else self-anchored.
    golden_ids = set(req.golden_crop_ids or [])
    golden_mask = np.array([crop.id in golden_ids for crop, _n, _li in rows], dtype=bool)
    if golden_mask.any():
        scores = anomaly_scores_vs_golden(emb, label_arr, golden_mask)
    else:
        scores = anomaly_scores_per_class(emb, label_arr)

    points: list[PlotPoint] = []
    for i, (crop, label_name, li) in enumerate(rows):
        x, y = coords[i]
        points.append(
            PlotPoint(
                id=crop.id,
                x=float(x),
                y=float(y),
                cluster=li,
                url=crop.url,
                anomaly_score=float(scores[i]),
                label=label_name,
                is_golden=bool(golden_mask[i]),
                source_image_id=crop.source_image_id,
                instance_index=crop.instance_index,
                crop_annotation=crop.crop_annotation,
            )
        )

    return AnalyzeResponse(points=points, labels=label_names)


@router.post("/inference/analyze", response_model=AnalyzeResponse)
async def analyze_embeddings(req: AnalyzeRequest, db: Session = Depends(get_db)) -> AnalyzeResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _analyze_embeddings, req, db)
