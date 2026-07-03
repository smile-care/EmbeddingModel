"""Inter-class relation analysis in full-dim embedding space.

All metrics operate on L2-normalized embeddings (cosine geometry), never on the
2-D projection, so distances are faithful. Produces:

  * kNN cross-class confusion matrix (scale-free, interpretable)
  * class-centroid cosine similarity matrix
  * per-class quality (size, compactness, silhouette, nearest other class)
  * hierarchical clustering (dendrogram) of classes -> over-split signal
  * mislabel candidates (samples whose neighborhood is dominated by another class)
  * headline separability score
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors

#: silhouette is O(n^2); above this we estimate it on a random subsample.
_SILHOUETTE_MAX_N = 5000


class CropMeta(TypedDict):
    id: str
    url: str
    source_image_id: str | None
    instance_index: int | None


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norms + 1e-12)


def adaptive_k(n: int) -> int:
    """Neighborhood size ~1% of samples, clamped to [5, 15] and kept < n."""
    k = int(np.clip(round(0.01 * n), 5, 15))
    return max(1, min(k, n - 1))


def _class_centroids(emb: np.ndarray, label_arr: np.ndarray, n_classes: int) -> np.ndarray:
    d = emb.shape[1]
    cents = np.zeros((n_classes, d), dtype=np.float64)
    for c in range(n_classes):
        mask = label_arr == c
        if mask.any():
            cents[c] = emb[mask].mean(axis=0)
    return _l2_normalize(cents)


def _confusion_matrix(
    emb: np.ndarray, label_arr: np.ndarray, n_classes: int, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalized kNN cross-class confusion + per-sample cross-class fractions.

    ``conf[i][j]`` = share of class-i samples' neighbors that belong to class j.
    Returns ``(conf, neighbor_labels)`` where ``neighbor_labels`` is (n, k) the
    class of each sample's k nearest neighbors (self excluded).
    """
    n = emb.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
    nn.fit(emb)
    _dist, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]  # drop self (first neighbor)
    neighbor_labels = label_arr[idx]  # (n, k)

    conf = np.zeros((n_classes, n_classes), dtype=np.float64)
    for c in range(n_classes):
        mask = label_arr == c
        if not mask.any():
            continue
        nbrs = neighbor_labels[mask].ravel()
        counts = np.bincount(nbrs, minlength=n_classes).astype(np.float64)
        total = counts.sum()
        if total > 0:
            conf[c] = counts / total
    return conf, neighbor_labels


def _per_class_quality(
    emb: np.ndarray,
    label_arr: np.ndarray,
    centroids: np.ndarray,
    centroid_sim: np.ndarray,
    label_names: list[str],
) -> tuple[list[dict[str, Any]], float]:
    n = emb.shape[0]
    n_classes = len(label_names)

    # silhouette (cosine); subsample for very large n to bound O(n^2).
    sil_per_class = np.full(n_classes, float("nan"))
    mean_sil = float("nan")
    if n_classes >= 2 and n > n_classes:
        if n > _SILHOUETTE_MAX_N:
            rng = np.random.default_rng(42)
            sub = rng.choice(n, size=_SILHOUETTE_MAX_N, replace=False)
            emb_s, lab_s = emb[sub], label_arr[sub]
        else:
            emb_s, lab_s = emb, label_arr
        # silhouette needs >=2 labels present with >=1 sample each in the subsample
        if len(np.unique(lab_s)) >= 2:
            sil = silhouette_samples(emb_s, lab_s, metric="cosine")
            mean_sil = float(np.mean(sil))
            for c in range(n_classes):
                m = lab_s == c
                if m.any():
                    sil_per_class[c] = float(np.mean(sil[m]))

    out: list[dict[str, Any]] = []
    for c in range(n_classes):
        mask = label_arr == c
        count = int(mask.sum())
        if count:
            # compactness = mean cosine distance to own centroid (lower = tighter)
            sims = emb[mask] @ centroids[c]
            compactness = float(1.0 - np.mean(sims))
        else:
            compactness = float("nan")

        # nearest other class by centroid similarity
        row = centroid_sim[c].copy()
        row[c] = -np.inf
        nearest = int(np.argmax(row)) if n_classes > 1 else -1
        nearest_sim = float(row[nearest]) if nearest >= 0 else float("nan")

        out.append(
            {
                "label": label_names[c],
                "count": count,
                "compactness": compactness,
                "silhouette": None if np.isnan(sil_per_class[c]) else float(sil_per_class[c]),
                "nearestOtherLabel": label_names[nearest] if nearest >= 0 else None,
                "nearestOtherSim": None if np.isnan(nearest_sim) else nearest_sim,
            }
        )
    return out, mean_sil


def _linkage_tree(centroid_sim: np.ndarray, label_names: list[str]) -> dict[str, Any] | None:
    """Average-linkage hierarchical clustering of classes -> nested tree JSON.

    Distance = 1 - centroid cosine similarity. Low merge height between two
    classes means they are barely separable (over-split candidates).
    """
    n_classes = len(label_names)
    if n_classes < 2:
        return None
    dist = 1.0 - centroid_sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    iu = np.triu_indices(n_classes, k=1)
    condensed = np.clip(dist[iu], 0.0, None)
    Z = linkage(condensed, method="average")

    # Build nested tree from the linkage matrix.
    nodes: dict[int, dict[str, Any]] = {
        i: {"name": label_names[i], "height": 0.0, "children": []} for i in range(n_classes)
    }
    for m, (a, b, height, _count) in enumerate(Z):
        new_id = n_classes + m
        nodes[new_id] = {
            "name": None,
            "height": float(height),
            "children": [nodes[int(a)], nodes[int(b)]],
        }
    return nodes[n_classes + len(Z) - 1]


def _mislabel_candidates(
    neighbor_labels: np.ndarray,
    label_arr: np.ndarray,
    label_names: list[str],
    crop_meta: list[CropMeta],
    *,
    cross_threshold: float = 0.5,
    top_n: int = 300,
) -> list[dict[str, Any]]:
    """Samples whose neighborhood is dominated by a single *other* class."""
    n, k = neighbor_labels.shape
    n_classes = len(label_names)
    out: list[dict[str, Any]] = []
    for i in range(n):
        own = label_arr[i]
        counts = np.bincount(neighbor_labels[i], minlength=n_classes)
        cross_fraction = float(1.0 - counts[own] / k)
        if cross_fraction < cross_threshold:
            continue
        other = counts.copy()
        other[own] = -1
        suggested = int(np.argmax(other))
        # require the suggested class to actually dominate the neighborhood
        if counts[suggested] <= counts[own]:
            continue
        meta = crop_meta[i]
        out.append(
            {
                "cropId": meta["id"],
                "url": meta["url"],
                "sourceImageId": meta.get("source_image_id"),
                "instanceIndex": meta.get("instance_index"),
                "currentLabel": label_names[own],
                "suggestedLabel": label_names[suggested],
                "crossFraction": cross_fraction,
                "suggestedShare": float(counts[suggested] / k),
            }
        )
    out.sort(key=lambda r: (r["suggestedShare"], r["crossFraction"]), reverse=True)
    return out[:top_n]


def compute_relations(
    emb: np.ndarray,
    label_indices: np.ndarray,
    label_names: list[str],
    crop_meta: list[CropMeta],
    k: int | None = None,
) -> dict[str, Any]:
    """Full relation-analysis payload (see module docstring)."""
    emb = _l2_normalize(np.asarray(emb, dtype=np.float64))
    label_arr = np.asarray(label_indices, dtype=int)
    n = emb.shape[0]
    n_classes = len(label_names)

    counts = [int((label_arr == c).sum()) for c in range(n_classes)]

    if n < 2 or n_classes < 2:
        return {
            "labels": label_names,
            "counts": counts,
            "k": 0,
            "confusion": np.zeros((n_classes, n_classes)).tolist(),
            "centroidSim": np.eye(n_classes).tolist(),
            "perClass": [
                {
                    "label": label_names[c],
                    "count": counts[c],
                    "compactness": None,
                    "silhouette": None,
                    "nearestOtherLabel": None,
                    "nearestOtherSim": None,
                }
                for c in range(n_classes)
            ],
            "linkage": None,
            "mislabels": [],
            "headline": {
                "purity": None,
                "meanSilhouette": None,
                "nClasses": n_classes,
                "nSamples": n,
            },
        }

    k = k or adaptive_k(n)
    centroids = _class_centroids(emb, label_arr, n_classes)
    centroid_sim = np.clip(centroids @ centroids.T, -1.0, 1.0)
    conf, neighbor_labels = _confusion_matrix(emb, label_arr, n_classes, k)
    per_class, mean_sil = _per_class_quality(
        emb, label_arr, centroids, centroid_sim, label_names
    )
    tree = _linkage_tree(centroid_sim, label_names)
    mislabels = _mislabel_candidates(neighbor_labels, label_arr, label_names, crop_meta)

    # weighted kNN purity (diagonal weighted by class size)
    total = sum(counts)
    purity = float(sum(conf[c, c] * counts[c] for c in range(n_classes)) / total) if total else None

    return {
        "labels": label_names,
        "counts": counts,
        "k": int(k),
        "confusion": conf.tolist(),
        "centroidSim": centroid_sim.tolist(),
        "perClass": per_class,
        "linkage": tree,
        "mislabels": mislabels,
        "headline": {
            "purity": purity,
            "meanSilhouette": None if np.isnan(mean_sil) else float(mean_sil),
            "nClasses": n_classes,
            "nSamples": n,
        },
    }
