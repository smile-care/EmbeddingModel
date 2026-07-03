from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from data_cluster.app.config import get_settings
from data_cluster.app.database import SessionLocal, init_db
from data_cluster.app.routers.datasets import router as datasets_router
from data_cluster.app.routers.experiments import router as experiments_router
from data_cluster.app.routers.inference import router as inference_router
from data_cluster.app.services.storage import ensure_upload_root


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    # On startup, any experiment still marked "Running" means the process was
    # killed mid-training. Reset them so users can retry or delete them.
    _reset_stale_running_experiments()
    _sweep_orphan_embedding_cache()
    yield


def _sweep_orphan_embedding_cache() -> None:
    """Drop cached embedding npz files whose inference run no longer exists."""
    from data_cluster.app.models.db import InferenceRun
    from data_cluster.app.services.embedding_cache import sweep_orphans

    db = SessionLocal()
    try:
        valid = {r.id for r in db.query(InferenceRun.id).all()}
    finally:
        db.close()
    sweep_orphans(get_settings(), valid)


def _reset_stale_running_experiments() -> None:
    from data_cluster.app.models.db import Experiment
    db = SessionLocal()
    try:
        # 新模型用 run_status；同时兼容历史数据中遗留的 status == "Running"。
        stale = (
            db.query(Experiment)
            .filter((Experiment.run_status == "Running") | (Experiment.status == "Running"))
            .all()
        )
        for exp in stale:
            # 只标记「本次运行」为失败；上一次成功的结果字段保持不变。
            exp.run_status = "Failed"
            existing = exp.run_metrics or {}
            exp.run_metrics = {
                **existing,
                "stage": "failed",
                "error": "训练进程因后端重启而中断，请重新训练。",
            }
        if stale:
            db.commit()
    finally:
        db.close()


# StaticFiles requires the directory to exist before mount; lifespan runs too late.
# Resolve repo root so running this file from the IDE (any cwd) still uses ./data correctly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(_REPO_ROOT)
Path("data").mkdir(parents=True, exist_ok=True)
_settings = get_settings()
ensure_upload_root(_settings)

app = FastAPI(title="Data Cluster API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    GZipMiddleware,
    minimum_size=1024,
    compresslevel=6,
)

app.mount(_settings.static_mount_path, StaticFiles(directory=str(_settings.upload_dir)), name="static")

app.include_router(datasets_router, prefix="/api")
app.include_router(experiments_router, prefix="/api")
app.include_router(inference_router, prefix="/api")




if __name__ == "__main__":
    """Match run_backend.sh: PYTHONPATH=src, uvicorn with reload (import string required)."""
    here = Path(__file__).resolve()
    src_dir = here.parents[2]
    repo_root = here.parents[3]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    os.chdir(repo_root)
    prev = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(src_dir) if not prev else f"{src_dir}{os.pathsep}{prev}"

    import uvicorn

    host = os.getenv("DATA_CLUSTER_HOST", "0.0.0.0")
    port = int(os.getenv("DATA_CLUSTER_PORT", "8000"))
    workers = int(os.getenv("DATA_CLUSTER_WORKERS", "1"))
    log_level = os.getenv("DATA_CLUSTER_LOG_LEVEL", "info")
    access_log = os.getenv("DATA_CLUSTER_ACCESS_LOG", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    reload_enabled = True

    uvicorn.run(
        "data_cluster.app.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        reload_dirs=[str(src_dir)] if reload_enabled else None,
        workers=workers,
        log_level=log_level,
        access_log=access_log,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=30,
        backlog=2048,
    )
