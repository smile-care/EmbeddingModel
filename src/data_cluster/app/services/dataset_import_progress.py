"""Persist dataset import progress for frontend polling."""

import time

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

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
_RETRY_ATTEMPTS = 8
_RETRY_BASE_DELAY_SEC = 0.05


def _progress_for_stage(stage: str, sub_progress: float) -> float:
    start, end = _STAGE_RANGE.get(stage, (0.0, 100.0))
    sub = max(0.0, min(1.0, sub_progress))
    if stage == "done":
        return 100.0
    if stage == "failed":
        return start
    return start + (end - start) * sub


def _apply_import_progress(
    ds: Dataset,
    stage: str,
    *,
    sub_progress: float = 0.0,
    message: str | None = None,
) -> None:
    progress = _progress_for_stage(stage, sub_progress)
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


def _commit_with_retry(db: Session) -> None:
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            db.commit()
            return
        except OperationalError as exc:
            db.rollback()
            if "database is locked" not in str(exc).lower() or attempt == _RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_BASE_DELAY_SEC * (2**attempt))


def update_import_progress(
    dataset_id: str,
    stage: str,
    *,
    sub_progress: float = 0.0,
    message: str | None = None,
    force: bool = False,
    db: Session | None = None,
) -> None:
    now = time.monotonic()
    if not force and now - _last_commit.get(dataset_id, 0.0) < _THROTTLE_SEC:
        return
    _last_commit[dataset_id] = now

    if db is not None:
        ds = db.get(Dataset, dataset_id)
        if not ds:
            return
        _apply_import_progress(ds, stage, sub_progress=sub_progress, message=message)
        db.flush()
        return

    for attempt in range(_RETRY_ATTEMPTS):
        session = SessionLocal()
        try:
            ds = session.get(Dataset, dataset_id)
            if not ds:
                return
            _apply_import_progress(ds, stage, sub_progress=sub_progress, message=message)
            _commit_with_retry(session)
            return
        except OperationalError as exc:
            session.rollback()
            if "database is locked" not in str(exc).lower() or attempt == _RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_BASE_DELAY_SEC * (2**attempt))
        finally:
            session.close()


def clear_import_progress(dataset_id: str) -> None:
    _last_commit.pop(dataset_id, None)
