"""OSNN: nearest-neighbor distance ratio (Mendes Júnior et al., 2017)."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def access_test_labels(vault):
    return vault.labels()


def scores(x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``ratio = d(x, t) / d(x, u)``. ``t`` is the nearest neighbor, ``u`` the nearest of another class.

    The predicted class is the class of ``t``. Higher ratio is more unknown.
    """
    y_train = np.asarray(y_train).astype(str)
    classes = np.unique(y_train)
    distances = []
    for label in classes:
        model = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        model.fit(x_train[y_train == label])
        dist, _ = model.kneighbors(x_query)
        distances.append(dist.ravel())
    stack = np.vstack(distances)
    nearest = np.argmin(stack, axis=0)
    rows = np.arange(stack.shape[1])
    d_t = stack[nearest, rows]
    d_u = np.empty(stack.shape[1], dtype=float)
    for index in range(len(classes)):
        mask = nearest == index
        if not np.any(mask):
            continue
        others = np.delete(stack, index, axis=0)
        d_u[mask] = np.inf if others.size == 0 else others.min(axis=0)[mask]
    ratio = d_t / np.maximum(d_u, 1e-12)
    return ratio, classes[nearest]
