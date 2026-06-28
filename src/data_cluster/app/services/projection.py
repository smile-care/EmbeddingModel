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
    """Returns scores in [0, 100] per sample (higher = more anomalous)."""
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
