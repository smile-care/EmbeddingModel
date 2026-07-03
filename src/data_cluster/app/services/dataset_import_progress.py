"""Persist dataset import progress for frontend polling."""

import time

from data_cluster.app.database import SessionLocal
from data_cluster.app.models.db import Dataset

# stage -> (start_pct, end_pct) on the 0-100 server-side scale
_STAGE_RANGE: dict[str, tuple[float, float]] = {
    "uploading": (0.0, 5.0),
    "extracting": (5.0, 45.0),
    "persisting": (45.0, 55.0),
    "cropping": (55.0, 99.0),
    "saving": (5.0, 95.0),
    "done": (100.0, 100.0),
    "failed": (0.0, 0.0),
}

_last_commit: dict[str, float] = {}
_THROTTLE_SEC = 0.25


def _progress_for_stage(stage: str, sub_progress: float) -> float:
    start, end = _STAGE_RANGE.get(stage, (0.0, 100.0))
    sub = max(0.0, min(1.0, sub_progress))
    if stage == "done":
        return 100.0
    if stage == "failed":
        return start
    return start + (end - start) * sub


def update_import_progress(
    dataset_id: str,
    stage: str,
    *,
    sub_progress: float = 0.0,
    message: str | None = None,
    force: bool = False,
) -> None:
    now = time.monotonic()
    if not force and now - _last_commit.get(dataset_id, 0.0) < _THROTTLE_SEC:
        return
    _last_commit[dataset_id] = now

    progress = _progress_for_stage(stage, sub_progress)
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds:
            return
        ds.import_stage = stage
        ds.import_progress = progress
        if message is not None:
            ds.import_message = message
        if stage == "done":
            ds.status = "Ready"
            ds.import_progress = 100.0
        elif stage == "failed":
            ds.status = "Failed"
        else:
            ds.status = "Processing"
        db.commit()
    finally:
        db.close()


def clear_import_progress(dataset_id: str) -> None:
    _last_commit.pop(dataset_id, None)
