import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session, selectinload

from data_cluster.app.config import get_settings
from data_cluster.app.database import get_db
from data_cluster.app.models.db import Dataset, Experiment
from data_cluster.app.schemas.experiment import (ExperimentCreate, ExperimentDetail,
                                                 ExperimentSummary)
from data_cluster.app.services.storage import experiment_checkpoint_run_dir
from data_cluster.app.services.training import (get_experiment_sample_counts,
                                                normalize_experiment_metrics,
                                                prepare_experiment_samples, request_stop,
                                                run_training_job)

router = APIRouter(prefix="/experiments", tags=["experiments"])

# 专用线程池：训练是 CPU/GPU 密集型，单独隔离避免和 IO 线程争用
_training_executor = None


def _get_executor():
    """懒初始化独立 ThreadPoolExecutor 用于训练任务。"""
    global _training_executor
    if _training_executor is None:
        from concurrent.futures import ThreadPoolExecutor
        # 训练任务同时只需一个 worker，但保留2个以允许并发 stop/status 检查
        _training_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="training")
    return _training_executor


def _serialize_experiment(e: Experiment) -> ExperimentDetail:
    total_samples, train_samples, val_samples = get_experiment_sample_counts(e)
    return ExperimentDetail(
        id=e.id,
        name=e.name,
        model=e.model,
        dataset=e.dataset,
        status=e.status,
        duration=e.duration,
        accuracy=e.accuracy,
        progress=e.progress,
        created_at=e.created_at,
        dataset_id=e.dataset_id,
        config=e.config,
        metrics=normalize_experiment_metrics(e.metrics),
        checkpoint_path=e.checkpoint_path,
        sample_count=total_samples,
        train_sample_count=train_samples,
        val_sample_count=val_samples,
    )


def _serialize_summary(e: Experiment) -> ExperimentSummary:
    return ExperimentSummary(
        id=e.id,
        name=e.name,
        model=e.model,
        dataset=e.dataset,
        status=e.status,
        duration=e.duration,
        accuracy=e.accuracy,
        progress=e.progress,
        created_at=e.created_at,
    )


# ── 轮询热路径：只查询轻量字段，不加载 samples ──────────────────────────────────
def _query_experiment_light(db: Session, experiment_id: str) -> Experiment | None:
    """轻量查询，不 selectinload samples，专用于轮询进度接口。"""
    return db.query(Experiment).filter(Experiment.id == experiment_id).first()


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments(db: Session = Depends(get_db)) -> list[ExperimentSummary]:
    loop = asyncio.get_event_loop()

    def _query():
        return (
            db.query(Experiment)
            .options(selectinload(Experiment.samples))
            .order_by(Experiment.created_at.desc())
            .all()
        )

    rows = await loop.run_in_executor(None, _query)
    return [_serialize_summary(e) for e in rows]


@router.get("/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(experiment_id: str, db: Session = Depends(get_db)) -> ExperimentDetail:
    loop = asyncio.get_event_loop()

    def _query():
        return (
            db.query(Experiment)
            .options(selectinload(Experiment.samples))
            .filter(Experiment.id == experiment_id)
            .first()
        )

    e = await loop.run_in_executor(None, _query)
    if not e:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _serialize_experiment(e)


@router.post("", response_model=ExperimentDetail)
async def create_experiment(body: ExperimentCreate, db: Session = Depends(get_db)) -> ExperimentDetail:
    loop = asyncio.get_event_loop()

    def _create():
        dataset_name = body.dataset
        if body.dataset_id:
            ds = db.get(Dataset, body.dataset_id)
            if ds:
                dataset_name = ds.name
        exp = Experiment(
            name=body.name,
            model=body.model,
            dataset=dataset_name,
            dataset_id=body.dataset_id,
            status="Pending",
            duration="---",
            accuracy="---",
            progress=0.0,
            config=body.config,
            metrics=None,
        )
        db.add(exp)
        db.commit()
        db.refresh(exp)
        try:
            prepare_experiment_samples(db, exp, get_settings())
            db.commit()
            db.refresh(exp)
        except ValueError as exc:
            db.delete(exp)
            db.commit()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return (
            db.query(Experiment)
            .options(selectinload(Experiment.samples))
            .filter(Experiment.id == exp.id)
            .first()
        )

    exp = await loop.run_in_executor(None, _create)
    assert exp is not None
    return _serialize_experiment(exp)


@router.post("/{experiment_id}/stop")
async def stop_training(experiment_id: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    """Request a running training job to stop.

    Sets the in-process stop flag (if the trainer is still running in this
    process) AND immediately marks the experiment as Failed in the DB so the
    frontend unblocks even if the process was restarted.
    """
    loop = asyncio.get_event_loop()

    def _stop():
        exp = db.get(Experiment, experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        request_stop(experiment_id)
        if exp.status == "Running":
            existing = exp.metrics or {}
            exp.status = "Stopped"
            exp.metrics = {**existing, "stage": "stopped"}
            db.commit()

    await loop.run_in_executor(None, _stop)
    return {"stopping": True}


@router.post("/{experiment_id}/train")
async def start_training(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, bool]:
    loop = asyncio.get_event_loop()

    def _check():
        return db.get(Experiment, experiment_id)

    exp = await loop.run_in_executor(None, _check)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # 用独立 executor 运行训练，避免占用 uvicorn 默认线程池
    executor = _get_executor()
    loop.run_in_executor(executor, run_training_job, experiment_id)
    return {"started": True}


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    loop = asyncio.get_event_loop()

    def _delete():
        exp = db.get(Experiment, experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        if exp.status == "Running":
            request_stop(experiment_id)
        settings = get_settings()
        run_dir = experiment_checkpoint_run_dir(settings, exp.model, experiment_id)
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
        legacy = settings.checkpoints_dir / f"{experiment_id}.pt"
        if legacy.is_file():
            legacy.unlink()
        if exp.checkpoint_path:
            cp = Path(exp.checkpoint_path)
            if cp.is_file():
                cp.unlink()
        db.delete(exp)
        db.commit()

    await loop.run_in_executor(None, _delete)
    return {"success": True}
