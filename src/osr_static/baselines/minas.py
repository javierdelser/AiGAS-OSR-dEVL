"""MINAS-style novelty detection (Faria et al., DMKD 2016), batch OSR adaptation.

Each known class is summarized by ``k`` micro-clusters (K-Means). A test point
is assigned the label of the nearest micro-cluster centroid. The reject score is
the ratio of Euclidean distance to that centroid over the micro-cluster radius
(radius = ``factor`` times the within-cluster distance standard deviation).
Higher scores indicate unknown/novel examples (outside the micro-cluster shell).

Open-set rejection in this repository uses the paper rule ``Dist > radius`` i.e.
``score > 1`` (Algorithm 2), not an extra validation threshold on ``score``.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def access_test_labels(vault):
    return vault.labels()


def fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k_micro: int = 5,
    radius_factor: float = 2.0,
    random_state: int = 0,
) -> dict:
    y_train = np.asarray(y_train).astype(str)
    x_train = np.asarray(x_train, dtype=float)
    centroids: list[np.ndarray] = []
    radii: list[float] = []
    labels: list[str] = []
    for label in np.unique(y_train):
        own = x_train[y_train == label]
        if own.shape[0] == 0:
            continue
        k_use = max(1, min(int(k_micro), own.shape[0]))
        model = KMeans(n_clusters=k_use, n_init=10, random_state=int(random_state))
        model.fit(own)
        for cluster in range(k_use):
            mask = model.labels_ == cluster
            pts = own[mask] if np.any(mask) else own
            center = model.cluster_centers_[cluster]
            dist = np.linalg.norm(pts - center, axis=1)
            spread = float(dist.std(ddof=0)) if dist.size > 1 else float(dist.max() if dist.size else 1e-6)
            radius = float(radius_factor) * max(spread, 1e-6)
            centroids.append(center)
            radii.append(radius)
            labels.append(str(label))
    return {
        "centroids": np.asarray(centroids, dtype=float),
        "radii": np.asarray(radii, dtype=float),
        "labels": np.asarray(labels, dtype=object),
        "k_micro": int(k_micro),
        "radius_factor": float(radius_factor),
    }


def scores(model: dict, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_query = np.asarray(x_query, dtype=float)
    centers = model["centroids"]
    radii = model["radii"]
    labels = model["labels"]
    if centers.shape[0] == 0:
        n = x_query.shape[0]
        return np.ones(n, dtype=float), np.array(["0"] * n, dtype=object)
    diff = x_query[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    nearest = np.argmin(dist, axis=1)
    rows = np.arange(x_query.shape[0])
    d = dist[rows, nearest]
    r = radii[nearest]
    # Outside the micro-cluster shell (Algorithm 2): unknown; score > 1 ⇒ reject with tau ≈ 1.
    score = d / np.maximum(r, 1e-12)
    pred = labels[nearest]
    return score.astype(float), np.asarray([str(p) for p in pred], dtype=object)


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k_micro: int = 5,
    radius_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    return scores(fit(x_train, y_train, k_micro, radius_factor), x_query)
