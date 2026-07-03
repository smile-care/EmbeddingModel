import asyncio
import hashlib
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
                                                ProjectionOut, RelationsOut, UploadCategoryCreate,
                                                UploadCategoryOut, UploadCategoryPatch)
from data_cluster.app.services.embedding_cache import (delete_embeddings, load_embeddings,
                                                       save_embeddings)
from data_cluster.app.services.storage import url_to_fs_path
from data_cluster.dl.projection import (anomaly_scores_per_class, anomaly_scores_vs_golden,
                                        project_2d)
from data_cluster.dl.relations import CropMeta, compute_relations
from data_cluster.dl.config_resolve import (
    default_model_ids,
    list_default_models,
    resolve_default_model,
)
from data_cluster.dl.inference import (INDUSTRIAL_MODEL_ID, INDUSTRIAL_MODEL_NAME,
                                       compute_industrial_pretrained_embeddings,
                                       compute_raw_embeddings,
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
        class_ids=list(run.selected_class_ids or []),
        algorithm=run.algorithm,
        view_mode=run.view_mode,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _run_to_detail(run: InferenceRun, db: Session) -> InferenceRunDetail:
    s = _run_to_summary(run, db)
    base = s.model_dump(by_alias=False, mode="python")
    current_fp, stale = _analysis_stale_for_run(run, db)
    result_json = run.result_json
    cached_algos: list[str] = []
    if not stale and current_fp and run.dataset_mode == "existing" and run.dataset_id:
        cached_algos = _cached_algorithms_for_fingerprint(db, run.id, current_fp)
        algo = (run.algorithm or "tsne").lower()
        proj = _get_valid_projection(db, run.id, algo, current_fp)
        if proj:
            base_payload = dict(result_json) if isinstance(result_json, dict) else {}
            result_json = {
                **{k: v for k, v in base_payload.items() if k not in ("points", "labels", "fingerprint")},
                "labels": proj.labels,
                "points": proj.points,
                "fingerprint": current_fp,
            }
    elif stale and isinstance(result_json, dict):
        result_json = {k: v for k, v in result_json.items() if k not in ("points", "labels")}
    return InferenceRunDetail(
        **base,
        result_json=result_json,
        cached_algorithms=cached_algos,
        analysis_stale=stale,
        analysis_fingerprint=current_fp,
    )


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


def _dataset_crop_rows(
    ds: Dataset,
    class_ids: set[str] | None = None,
) -> list[tuple[CropImage, str, int]]:
    crop_rows = [
        crop
        for crop in ds.crop_images
        if crop.defect_class is not None
        and crop.url
        and (class_ids is None or crop.class_id in class_ids)
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
            selected_class_ids=list(body.class_ids or []) or None,
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
        if "classIds" in data:
            row.selected_class_ids = list(data["classIds"] or []) or None
        if "algorithm" in data and data["algorithm"] is not None:
            row.algorithm = str(data["algorithm"]).lower()
            if row.dataset_mode == "existing" and row.dataset_id:
                try:
                    req = _analyze_request_for_run(row)
                    rows, _ln, _la = _resolve_analysis_rows(req, db)
                    fp = _analysis_fingerprint(req, db, rows)
                    proj = _get_valid_projection(db, run_id, row.algorithm, fp)
                    if proj:
                        _apply_result_json(row, list(proj.labels), list(proj.points), fp)
                except HTTPException:
                    pass
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
        delete_embeddings(get_settings(), run_id)

    await loop.run_in_executor(None, _delete)
    return {"success": True}


@router.delete("/inference/runs/{run_id}/embedding-cache")
async def delete_inference_embedding_cache(run_id: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    """Remove the on-disk embedding cache for a run (does not delete the run record)."""
    loop = asyncio.get_event_loop()

    def _delete_cache():
        row = db.get(InferenceRun, run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Inference run not found")
        delete_embeddings(get_settings(), run_id)
        _delete_run_projections(db, run_id)
        if isinstance(row.result_json, dict):
            row.result_json = {
                k: v
                for k, v in row.result_json.items()
                if k not in ("points", "labels", "fingerprint")
            }
            db.commit()

    await loop.run_in_executor(None, _delete_cache)
    return {"success": True}


def _delete_run_projections(db: Session, run_id: str) -> None:
    db.query(InferenceRunProjection).filter(InferenceRunProjection.run_id == run_id).delete()


def _purge_stale_projections(db: Session, run_id: str, fingerprint: str) -> None:
    db.query(InferenceRunProjection).filter(
        InferenceRunProjection.run_id == run_id,
        InferenceRunProjection.fingerprint != fingerprint,
    ).delete(synchronize_session=False)


def _get_valid_projection(
    db: Session, run_id: str, algorithm: str, fingerprint: str
) -> InferenceRunProjection | None:
    proj = (
        db.query(InferenceRunProjection)
        .filter(
            InferenceRunProjection.run_id == run_id,
            InferenceRunProjection.algorithm == algorithm.lower(),
        )
        .first()
    )
    if not proj or not proj.fingerprint or proj.fingerprint != fingerprint:
        return None
    return proj


def _cached_algorithms_for_fingerprint(db: Session, run_id: str, fingerprint: str) -> list[str]:
    rows = (
        db.query(InferenceRunProjection.algorithm)
        .filter(
            InferenceRunProjection.run_id == run_id,
            InferenceRunProjection.fingerprint == fingerprint,
        )
        .all()
    )
    return [r[0] for r in rows]


def _apply_result_json(
    row: InferenceRun, labels: list, points: list, fingerprint: str
) -> None:
    row.result_json = {"labels": labels, "points": points, "fingerprint": fingerprint}


def _upsert_projection(
    db: Session,
    run_id: str,
    algorithm: str,
    labels: list,
    points: list,
    fingerprint: str,
) -> None:
    algo = algorithm.lower()
    existing = (
        db.query(InferenceRunProjection)
        .filter(InferenceRunProjection.run_id == run_id, InferenceRunProjection.algorithm == algo)
        .first()
    )
    if existing:
        existing.labels = labels
        existing.points = points
        existing.fingerprint = fingerprint
    else:
        db.add(
            InferenceRunProjection(
                run_id=run_id,
                algorithm=algo,
                labels=labels,
                points=points,
                fingerprint=fingerprint,
            )
        )


def _try_serve_cached_analysis(
    db: Session, row: InferenceRun, req: AnalyzeRequest, run_id: str
) -> bool:
    """Return cached scatter for the current algorithm when fingerprint still matches."""
    rows, _label_names, _label_arr = _resolve_analysis_rows(req, db)
    fp = _analysis_fingerprint(req, db, rows)
    proj = _get_valid_projection(db, run_id, row.algorithm.lower(), fp)
    if not proj:
        return False
    _apply_result_json(row, list(proj.labels), list(proj.points), fp)
    row.status = "Completed"
    db.commit()
    db.refresh(row)
    return True


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
                class_ids=list(row.selected_class_ids or []),
            )

            def _maybe_cached() -> bool:
                return _try_serve_cached_analysis(db, row, req, run_id)

            if await loop.run_in_executor(None, _maybe_cached):
                return _run_to_detail(row, db)

            resp = await loop.run_in_executor(None, _analyze_embeddings, req, db, run_id)
            labels = list(resp.labels)
            points = [p.model_dump(by_alias=True) for p in resp.points]

            def _save_result():
                rows, _label_names, _label_arr = _resolve_analysis_rows(req, db)
                fp = _analysis_fingerprint(req, db, rows)
                _apply_result_json(row, labels, points, fp)
                row.status = "Completed"
                algo = row.algorithm.lower()
                _upsert_projection(db, run_id, algo, labels, points, fp)
                _purge_stale_projections(db, run_id, fp)
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
                _upsert_projection(db, run_id, row.algorithm.lower(), labels, points, "upload")
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
        row = db.get(InferenceRun, run_id)
        if not row or row.dataset_mode != "existing" or not row.dataset_id:
            return None
        req = AnalyzeRequest(
            dataset_id=row.dataset_id,
            method=algo,
            experiment_id=row.model_id,
            model_id=row.model_id,
            golden_crop_ids=list(row.golden_crop_ids or []),
            class_ids=list(row.selected_class_ids or []),
        )
        rows, _ln, _la = _resolve_analysis_rows(req, db)
        fp = _analysis_fingerprint(req, db, rows)
        return _get_valid_projection(db, run_id, algo, fp)

    proj = await loop.run_in_executor(None, _query)
    if not proj:
        raise HTTPException(
            status_code=404,
            detail=f"No valid cached projection for algorithm '{algo}' (missing or stale)",
        )
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
        class_ids=list(row.selected_class_ids or []),
    )

    def _compute_or_load():
        rows, _label_names, _label_arr = _resolve_analysis_rows(req, db)
        fp = _analysis_fingerprint(req, db, rows)
        cached = _get_valid_projection(db, run_id, algo, fp)
        if cached:
            labels = list(cached.labels)
            points = [dict(p) for p in cached.points]
            return labels, points, fp, True
        resp = _analyze_embeddings(req, db, run_id)
        labels = list(resp.labels)
        points = [p.model_dump(by_alias=True) for p in resp.points]
        return labels, points, fp, False

    labels, points, fp, from_cache = await loop.run_in_executor(None, _compute_or_load)

    def _save():
        if not from_cache:
            _upsert_projection(db, run_id, algo, labels, points, fp)
            _purge_stale_projections(db, run_id, fp)
        if row.algorithm.lower() == algo:
            _apply_result_json(row, labels, points, fp)
        db.commit()

    await loop.run_in_executor(None, _save)
    return ProjectionOut(algorithm=algo, labels=labels, points=points)


@router.get("/models", response_model=list[ModelInfo])
async def list_models(db: Session = Depends(get_db)) -> list[ModelInfo]:
    """Return configured DINOv3 RAW models, industrial pretrained, then completed experiments."""
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None,
        lambda: db.query(Experiment)
        .filter(Experiment.status == "Completed")
        .order_by(Experiment.created_at.desc())
        .all()
    )
    models = [
        ModelInfo(id=m["id"], name=m["name"], type="Pretrained")
        for m in list_default_models()
    ]
    models.append(
        ModelInfo(id=INDUSTRIAL_MODEL_ID, name=INDUSTRIAL_MODEL_NAME, type="Pretrained")
    )
    models.extend(ModelInfo(id=e.id, name=e.name, type="Trained") for e in rows)
    return models


def _resolve_analysis_rows(
    req: AnalyzeRequest, db: Session
) -> tuple[list[tuple[CropImage, str, int]], list[str], np.ndarray]:
    """Validate dataset/classes and build the ordered crop rows (no GPU work)."""
    ds = (
        db.query(Dataset)
        .options(selectinload(Dataset.crop_images).selectinload(CropImage.defect_class))
        .filter(Dataset.id == req.dataset_id)
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if req.class_ids and len(req.class_ids) < 2:
        raise HTTPException(status_code=422, detail="至少需要选择 2 个类别进行分析。")

    class_id_set = set(req.class_ids) if req.class_ids else None
    rows = _dataset_crop_rows(ds, class_id_set)
    label_names = sorted({label_name for _crop, label_name, _li in rows})

    if len(label_names) < 2:
        raise HTTPException(
            status_code=422,
            detail="至少需要 2 个类别且每个类别有可用 crop 才能进行分析。",
        )
    if not rows:
        raise HTTPException(
            status_code=422,
            detail="当前数据集没有可用于分析的 crop 图片，请先生成裁剪图。",
        )

    label_arr = np.array([li for _crop, _name, li in rows], dtype=int)
    return rows, label_names, label_arr


def _compute_run_embeddings(
    req: AnalyzeRequest, db: Session, rows: list[tuple[CropImage, str, int]]
) -> np.ndarray:
    """Run the (GPU) forward pass for the resolved rows."""
    settings = get_settings()
    eid = req.experiment_id or req.model_id
    if not eid:
        raise HTTPException(status_code=422, detail="请选择模型。")

    paths = [_crop_fs_path(settings, crop) for crop, _n, _li in rows]
    mask_paths = [_crop_mask_fs_path(settings, crop) for crop, _n, _li in rows]

    if eid in default_model_ids():
        return compute_raw_embeddings(model_id=eid, image_paths=paths, mask_paths=mask_paths)
    if eid == INDUSTRIAL_MODEL_ID:
        return compute_industrial_pretrained_embeddings(image_paths=paths, mask_paths=mask_paths)

    exp = db.get(Experiment, eid)
    if not exp:
        raise HTTPException(status_code=404, detail=f"未知模型: {eid}")
    checkpoint = exp.checkpoint_path
    if not checkpoint:
        raise HTTPException(status_code=422, detail="所选实验没有 checkpoint_path，无法进行推理。")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=422, detail=f"模型 checkpoint 不存在: {checkpoint_path}")
    return compute_supcon_embeddings(
        checkpoint_path=checkpoint_path,
        image_paths=paths,
        mask_paths=mask_paths,
        exp_config=exp.config if isinstance(exp.config, dict) else None,
    )


def _path_mtime_token(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "0"
    return str(path.stat().st_mtime_ns)


def _model_cache_key(req: AnalyzeRequest, db: Session) -> str:
    """Stable identifier for the model producing embeddings (includes checkpoint mtime)."""
    eid = str(req.experiment_id or req.model_id or "")
    if not eid:
        return ""
    if eid in default_model_ids():
        try:
            _backbone, _bb_cfg, pretrained_path = resolve_default_model(eid)
            ckpt = Path(pretrained_path) if pretrained_path else None
            return f"{eid}:{_path_mtime_token(ckpt)}"
        except ValueError:
            return eid
    if eid == INDUSTRIAL_MODEL_ID:
        return eid
    exp = db.get(Experiment, eid)
    if not exp or not exp.checkpoint_path:
        return eid
    return f"{eid}:{_path_mtime_token(Path(exp.checkpoint_path))}"


def _data_fingerprint(settings, rows: list[tuple[CropImage, str, int]]) -> str:
    parts: list[str] = []
    for crop, _n, _li in rows:
        img = _crop_fs_path(settings, crop)
        mask = _crop_mask_fs_path(settings, crop)
        parts.append(f"{crop.id}:{_path_mtime_token(img)}:{_path_mtime_token(mask)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _config_fingerprint(req: AnalyzeRequest) -> str:
    classes = ",".join(sorted(req.class_ids or []))
    golden = ",".join(sorted(req.golden_crop_ids or []))
    return f"{req.dataset_id}|{classes}|{golden}"


def _analysis_fingerprint(
    req: AnalyzeRequest,
    db: Session,
    rows: list[tuple[CropImage, str, int]],
) -> str:
    settings = get_settings()
    raw = "|".join(
        (
            _model_cache_key(req, db),
            _data_fingerprint(settings, rows),
            _config_fingerprint(req),
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _stored_analysis_fingerprint(run: InferenceRun) -> str | None:
    payload = run.result_json
    if not isinstance(payload, dict):
        return None
    fp = payload.get("fingerprint")
    return str(fp) if fp else None


def _analyze_request_for_run(run: InferenceRun) -> AnalyzeRequest:
    return AnalyzeRequest(
        dataset_id=run.dataset_id or "",
        method=run.algorithm.lower() if run.algorithm else "tsne",
        experiment_id=run.model_id,
        model_id=run.model_id,
        golden_crop_ids=list(run.golden_crop_ids or []),
        class_ids=list(run.selected_class_ids or []),
    )


def _analysis_stale_for_run(run: InferenceRun, db: Session) -> tuple[str | None, bool]:
    """Return (current_fingerprint, is_stale) for an existing-dataset run."""
    if run.dataset_mode != "existing" or not run.dataset_id:
        return None, False
    stored = _stored_analysis_fingerprint(run)
    has_points = bool(
        isinstance(run.result_json, dict) and run.result_json.get("points")
    )
    if not stored and not has_points:
        return None, False
    try:
        req = _analyze_request_for_run(run)
        rows, _label_names, _label_arr = _resolve_analysis_rows(req, db)
        current = _analysis_fingerprint(req, db, rows)
    except HTTPException:
        return stored, bool(stored or has_points)
    stale = stored != current if stored else bool(has_points)
    return current, stale


def _load_or_compute_embeddings(
    req: AnalyzeRequest,
    db: Session,
    rows: list[tuple[CropImage, str, int]],
    label_names: list[str],
    label_arr: np.ndarray,
    run_id: str | None,
) -> np.ndarray:
    """Return embeddings for rows, reusing the per-run npz cache when it matches.

    The cache is keyed by run and validated against a content fingerprint (model
    checkpoint, crop file mtimes, class/golden filters) so stale data is
    recomputed instead of silently reused.
    """
    settings = get_settings()
    crop_ids = [crop.id for crop, _n, _li in rows]
    fingerprint = _analysis_fingerprint(req, db, rows)

    if run_id:
        cached = load_embeddings(settings, run_id)
        if (
            cached is not None
            and cached.get("fingerprint") == fingerprint
            and cached["emb"].shape[0] == len(crop_ids)
        ):
            return cached["emb"]

    emb = _compute_run_embeddings(req, db, rows)
    if run_id:
        save_embeddings(
            settings,
            run_id,
            emb=emb,
            crop_ids=crop_ids,
            label_indices=label_arr,
            label_names=label_names,
            fingerprint=fingerprint,
        )
    return emb


def _analyze_embeddings(
    req: AnalyzeRequest,
    db: Session,
    run_id: str | None = None,
) -> AnalyzeResponse:
    rows, label_names, label_arr = _resolve_analysis_rows(req, db)
    emb = _load_or_compute_embeddings(req, db, rows, label_names, label_arr, run_id)

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


def _compute_run_relations(run_id: str, db: Session, k: int | None) -> RelationsOut:
    row = db.get(InferenceRun, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Inference run not found")
    if row.dataset_mode != "existing" or not row.dataset_id:
        raise HTTPException(
            status_code=400, detail="关系分析仅支持 existing 数据集模式的推理任务。"
        )

    req = AnalyzeRequest(
        dataset_id=row.dataset_id,
        method=row.algorithm.lower() if row.algorithm else "tsne",
        experiment_id=row.model_id,
        model_id=row.model_id,
        golden_crop_ids=list(row.golden_crop_ids or []),
        class_ids=list(row.selected_class_ids or []),
    )
    rows, label_names, label_arr = _resolve_analysis_rows(req, db)
    emb = _load_or_compute_embeddings(req, db, rows, label_names, label_arr, run_id)

    crop_meta: list[CropMeta] = [
        {
            "id": crop.id,
            "url": crop.url,
            "source_image_id": crop.source_image_id,
            "instance_index": crop.instance_index,
        }
        for crop, _n, _li in rows
    ]
    payload = compute_relations(emb, label_arr, label_names, crop_meta, k=k)
    return RelationsOut(**payload)


@router.post("/inference/analyze", response_model=AnalyzeResponse)
async def analyze_embeddings(req: AnalyzeRequest, db: Session = Depends(get_db)) -> AnalyzeResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _analyze_embeddings, req, db)


@router.get("/inference/runs/{run_id}/relations", response_model=RelationsOut)
async def get_run_relations(
    run_id: str,
    k: int | None = None,
    db: Session = Depends(get_db),
) -> RelationsOut:
    """Inter-class relation analysis for a completed run (uses cached embeddings)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _compute_run_relations, run_id, db, k)
