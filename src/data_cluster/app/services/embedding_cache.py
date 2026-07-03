"""Regenerable per-run full-dim embedding cache for relation analysis.

Embeddings are the expensive (GPU) part of an inference run. Persisting them as
a small ``.npz`` per run lets the relation view and algorithm switching reuse the
forward pass instead of recomputing it. The cache is purely derived data: if a
file is missing it is transparently regenerated, so cleanup can be best-effort.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_cluster.app.config import Settings


def _cache_dir(settings: Settings) -> Path:
    d = settings.inference_cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(settings: Settings, run_id: str) -> Path:
    return _cache_dir(settings) / f"{run_id}.npz"


def save_embeddings(
    settings: Settings,
    run_id: str,
    *,
    emb: np.ndarray,
    crop_ids: list[str],
    label_indices: np.ndarray,
    label_names: list[str],
    model_key: str,
) -> None:
    """Persist (overwrite) the embedding cache for a run atomically.

    ``model_key`` identifies which model produced the embeddings so a cache from
    a different model is not silently reused when the run's model changes.
    """
    out = cache_path(settings, run_id)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        emb=np.asarray(emb, dtype=np.float32),
        crop_ids=np.asarray(crop_ids, dtype=object),
        label_indices=np.asarray(label_indices, dtype=np.int64),
        label_names=np.asarray(label_names, dtype=object),
        model_key=np.asarray(str(model_key), dtype=object),
    )
    tmp.replace(out)


def load_embeddings(settings: Settings, run_id: str) -> dict | None:
    """Return ``{emb, crop_ids, label_indices, label_names, model_key}`` or ``None``."""
    path = cache_path(settings, run_id)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            return {
                "emb": data["emb"],
                "crop_ids": [str(x) for x in data["crop_ids"].tolist()],
                "label_indices": data["label_indices"],
                "label_names": [str(x) for x in data["label_names"].tolist()],
                # Older caches predate model_key -> None forces a recompute.
                "model_key": str(data["model_key"]) if "model_key" in data else None,
            }
    except Exception:
        return None


def delete_embeddings(settings: Settings, run_id: str) -> None:
    """Best-effort removal of a run's cache file."""
    try:
        cache_path(settings, run_id).unlink(missing_ok=True)
    except OSError:
        pass


def sweep_orphans(settings: Settings, valid_run_ids: set[str]) -> int:
    """Delete cache files whose run no longer exists. Returns count removed."""
    d = settings.inference_cache_dir
    if not d.is_dir():
        return 0
    removed = 0
    for f in d.glob("*.npz"):
        if f.stem not in valid_run_ids:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed
