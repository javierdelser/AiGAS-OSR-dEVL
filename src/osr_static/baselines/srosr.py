"""Sparse-representation OSR via a class PCA residual (Zhang and Patel, 2017).

Each known class keeps the principal subspace that holds ``variance_keep`` of
that class's training variance. The predicted class is the one with the
smallest reconstruction residual. The reject score is that residual.
"""

from __future__ import annotations

import numpy as np


def access_test_labels(vault):
    return vault.labels()


def fit(x_train: np.ndarray, y_train: np.ndarray, variance_keep: float = 0.95) -> dict:
    y_train = np.asarray(y_train).astype(str)
    x_train = np.asarray(x_train, dtype=float)
    labels = [str(v) for v in np.unique(y_train)]
    subspaces = {}
    for label in labels:
        own = x_train[y_train == label]
        if own.shape[0] == 0:
            continue
        mean = own.mean(axis=0)
        centered = own - mean
        if own.shape[0] == 1 or not np.isfinite(centered).all():
            subspaces[label] = {"mean": mean, "basis": np.zeros((own.shape[1], 1))}
            continue
        _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        energy = singular ** 2
        total = float(energy.sum())
        if total <= 0:
            k = 1
        else:
            k = int(np.searchsorted(np.cumsum(energy) / total, float(variance_keep)) + 1)
            k = max(1, min(k, vt.shape[0], own.shape[1] - 1 if own.shape[1] > 1 else 1))
        subspaces[label] = {"mean": mean, "basis": vt[:k].T}
    return {"subspaces": subspaces, "labels": labels, "variance_keep": float(variance_keep)}


def scores(model: dict, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_query = np.asarray(x_query, dtype=float)
    labels = model["labels"]
    residuals = np.empty((x_query.shape[0], len(labels)), dtype=float)
    for index, label in enumerate(labels):
        mean = model["subspaces"][label]["mean"]
        basis = model["subspaces"][label]["basis"]
        centered = x_query - mean
        projected = (centered @ basis) @ basis.T
        residuals[:, index] = np.linalg.norm(centered - projected, axis=1)
    chosen = np.argmin(residuals, axis=1)
    pred = np.array(labels, dtype=object)[chosen]
    return residuals[np.arange(x_query.shape[0]), chosen], pred


def fit_predict(x_train, y_train, x_query, variance_keep: float = 0.95):
    return scores(fit(x_train, y_train, variance_keep), x_query)
