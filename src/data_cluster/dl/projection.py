from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ProjectionMethod = Literal["tsne", "umap", "pca"]


def project_2d(
    embeddings: np.ndarray,
    method: ProjectionMethod,
    random_state: int = 42,
) -> np.ndarray:
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    if n < 3:
        pca = PCA(n_components=min(2, n), random_state=random_state)
        return pca.fit_transform(embeddings)

    if method == "pca":
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(embeddings)

    if method == "umap":
        try:
            import umap

            n_neighbors = max(2, min(15, n - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=random_state,
            )
            return reducer.fit_transform(embeddings)
        except Exception:
            pca = PCA(n_components=2, random_state=random_state)
            return pca.fit_transform(embeddings)

    # tsne
    if n > 50:
        pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=random_state)
        emb = pca.fit_transform(embeddings)
    else:
        emb = embeddings
    perplexity = max(2.0, min(30.0, float(n - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(emb)


def anomaly_scores_per_class(embeddings: np.ndarray, label_indices: np.ndarray) -> np.ndarray:
    """Returns scores in [0, 100] per sample (higher = more anomalous).

    Self-anchored baseline: distance to the per-class centroid, min-max normalized
    *within* each class. Used when no golden reference samples are provided.
    """
    n = embeddings.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    if n == 0:
        return scores

    for c in np.unique(label_indices):
        mask = label_indices == c
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        sub = embeddings[mask]
        centroid = sub.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(sub - centroid, axis=1)
        if len(dist) == 1:
            scores[idx] = 50.0
            continue
        d_min, d_max = dist.min(), dist.max()
        if d_max - d_min < 1e-9:
            norm = np.full_like(dist, 50.0)
        else:
            norm = (dist - d_min) / (d_max - d_min) * 100.0
        scores[idx] = norm
    return scores


def _cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance (1 - cos) between rows of ``a`` and ``b``.

    Inputs are L2-normalized defensively so the result is robust even if the
    upstream embeddings were not already unit-norm.
    """
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = np.clip(a @ b.T, -1.0, 1.0)
    return 1.0 - sim


def _knn_mean(dist_rows: np.ndarray, k: int) -> np.ndarray:
    """Mean of the ``k`` smallest distances in each row."""
    k = max(1, min(k, dist_rows.shape[1]))
    if k >= dist_rows.shape[1]:
        return dist_rows.mean(axis=1)
    part = np.partition(dist_rows, k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


def anomaly_scores_vs_golden(
    embeddings: np.ndarray,
    label_indices: np.ndarray,
    golden_mask: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Golden-anchored anomaly scores in [0, 100] (higher = more anomalous).

    For each class that has at least one golden sample, a sample's raw anomaly is
    the mean cosine distance to its ``k`` nearest golden samples of the same class
    (kNN handles multi-modal golden sets better than a single centroid). The raw
    distance is then calibrated by the golden set's *own* internal spread so the
    0-100 scale is comparable across classes/datasets:

        ``score = 100 * (1 - exp(-d / r_c))``

    where ``r_c`` is the 90th-percentile of the golden samples' leave-one-out kNN
    distances (a robust "normal radius"). Golden samples themselves are pinned to
    0. Classes without golden fall back to :func:`anomaly_scores_per_class`.
    """
    n = embeddings.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    if n == 0:
        return scores

    golden_mask = np.asarray(golden_mask, dtype=bool)
    fallback = anomaly_scores_per_class(embeddings, label_indices)

    for c in np.unique(label_indices):
        cls_mask = label_indices == c
        cls_idx = np.where(cls_mask)[0]
        if len(cls_idx) == 0:
            continue
        g_idx = np.where(cls_mask & golden_mask)[0]
        if len(g_idx) == 0:
            scores[cls_idx] = fallback[cls_idx]
            continue

        golden_emb = embeddings[g_idx]
        cls_emb = embeddings[cls_idx]

        # raw distance of every class sample to its nearest golden of the same class
        d_to_golden = _knn_mean(_cosine_distance_matrix(cls_emb, golden_emb), k)

        # calibrate by golden's internal spread (leave-one-out kNN among golden)
        if len(g_idx) >= 2:
            gg = _cosine_distance_matrix(golden_emb, golden_emb)
            np.fill_diagonal(gg, np.inf)
            # cap k to exclude the self (inf diagonal) from each row's neighbourhood
            golden_loo = _knn_mean(gg, min(k, len(g_idx) - 1))
            finite = golden_loo[np.isfinite(golden_loo)]
            r_c = float(np.percentile(finite, 90)) if finite.size else float(np.median(d_to_golden))
        else:
            # single golden — no internal spread; use class median as the scale
            r_c = float(np.median(d_to_golden))
        r_c = max(r_c, 1e-6)

        cls_scores = 100.0 * (1.0 - np.exp(-d_to_golden / r_c))
        scores[cls_idx] = np.clip(cls_scores, 0.0, 100.0)

    # golden samples are the reference — pin them to 0
    scores[golden_mask] = 0.0
    return scores
