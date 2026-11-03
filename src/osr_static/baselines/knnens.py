"""KNNENS hypersphere ensemble (Zhang et al., IEEE TNNLS 2023), batch OSR adaptation.

Training follows Algorithm 1: ``b`` bootstrap bags, each with ``a`` instances per class;
hypersphere radius at center ``c`` is the mean distance to the ``L`` nearest same-class
neighbors (Eq. 1). Validation follows Eq. (5): if ``||x-c||/r(c) > t`` for the nearest
hypersphere the bag votes **NC**; otherwise the bag votes ``argmin_k dist(x, D_i(k))``.
Forest output is majority vote (Eq. 4). Buffer / model update (Algorithms 2–3) omitted.

Open-set rejection uses the paper NC label (mapped to ``UUC`` in the benchmark);
only the internal ratio threshold ``t`` is tuned on validation, not a second ``tau``.
"""

from __future__ import annotations

import numpy as np


def access_test_labels(vault):
    return vault.labels()


def _radius(pool: np.ndarray, center: np.ndarray, L: int) -> float:
    dist = np.linalg.norm(pool - center, axis=1)
    dist = np.sort(dist)
    if dist.size > 1:
        dist = dist[1:]
    take = min(max(1, int(L)), dist.size)
    return max(float(dist[:take].mean()) if take else 1e-6, 1e-6)


def _hyperspheres_for_bag(di_points: np.ndarray, di_labels: np.ndarray, L: int) -> dict:
    centers: list[np.ndarray] = []
    radii: list[float] = []
    labels: list[str] = []
    for label in np.unique(di_labels):
        pool = di_points[di_labels == label]
        for pt in pool:
            centers.append(pt)
            radii.append(_radius(pool, pt, L))
            labels.append(str(label))
    return {
        "centers": np.asarray(centers, dtype=float),
        "radii": np.asarray(radii, dtype=float),
        "labels": np.asarray(labels, dtype=object),
    }


def _bag(x_train: np.ndarray, y_train: np.ndarray, a: int, L: int, rng: np.random.Generator) -> dict:
    classes = np.unique(y_train)
    chunks: list[np.ndarray] = []
    chunk_labels: list[np.ndarray] = []
    for label in classes:
        pool = x_train[y_train == label]
        if pool.shape[0] == 0:
            continue
        take = min(int(a), pool.shape[0])
        idx = rng.choice(pool.shape[0], size=take, replace=False)
        chunks.append(pool[idx])
        chunk_labels.append(np.full(take, str(label), dtype=object))
    if not chunks:
        return _hyperspheres_for_bag(x_train, y_train, L)
    di_points = np.vstack(chunks)
    di_labels = np.concatenate(chunk_labels)
    built = _hyperspheres_for_bag(di_points, di_labels, L)
    return built


def fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    b: int = 5,
    a: int = 5,
    L: int = 5,
    seed: int = 0,
) -> dict:
    y_train = np.asarray(y_train).astype(str)
    x_train = np.asarray(x_train, dtype=float)
    rng = np.random.default_rng(int(seed))
    bags = [_bag(x_train, y_train, a, L, rng) for _ in range(max(1, int(b)))]
    return {"bags": bags, "b": int(b), "a": int(a), "L": int(L)}


def _bag_output(bag: dict, x_query: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = bag["centers"]
    radii = bag["radii"]
    labels = bag["labels"]
    diff = x_query[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    ratios = dist / radii[None, :]
    nearest = np.argmin(dist, axis=1)
    rows = np.arange(x_query.shape[0])
    min_ratio = ratios[rows, nearest]
    novel = min_ratio > float(threshold)
    pred = np.empty(x_query.shape[0], dtype=object)
    classes = np.unique(labels)
    for i in range(x_query.shape[0]):
        if novel[i]:
            pred[i] = "NC"
            continue
        best_k = classes[0]
        best_d = float("inf")
        for label in classes:
            mask = labels == label
            d = float(dist[i, mask].min())
            if d < best_d:
                best_d = d
                best_k = str(label)
        pred[i] = best_k
    return min_ratio.astype(float), pred, novel


def scores(model: dict, x_query: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    x_query = np.asarray(x_query, dtype=float)
    ratios: list[np.ndarray] = []
    votes: list[np.ndarray] = []
    for bag in model["bags"]:
        ratio, pred, _ = _bag_output(bag, x_query, threshold)
        ratios.append(ratio)
        votes.append(pred)
    mean_ratio = np.mean(np.vstack(ratios), axis=0)
    stack = np.stack(votes, axis=0)
    final = []
    for col in range(stack.shape[1]):
        column = stack[:, col]
        nc = int(np.sum(column == "NC"))
        if nc > stack.shape[0] / 2:
            final.append("NC")
            continue
        known = column[column != "NC"]
        if known.size == 0:
            final.append("NC")
            continue
        uniq, counts = np.unique(known, return_counts=True)
        final.append(str(uniq[int(np.argmax(counts))]))
    pred_out = np.asarray(final, dtype=object)
    return mean_ratio, np.asarray([str(p) for p in pred_out], dtype=object)


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    b: int = 5,
    a: int = 5,
    L: int = 5,
    threshold: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    model = fit(x_train, y_train, b, a, L)
    return scores(model, x_query, threshold)
