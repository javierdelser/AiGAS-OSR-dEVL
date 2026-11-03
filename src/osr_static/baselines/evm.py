"""A compact Extreme Value Machine (Rudd et al.). Set-cover reduction is not used.

Each training point keeps a Weibull fitted to the distances of its nearest
other-class neighbors. The class score is the maximum inclusion probability.
Reject when that maximum is below ``delta``. The stored score is higher for
more unknown: ``1 - max class score``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import weibull_min
from sklearn.neighbors import NearestNeighbors


def access_test_labels(vault):
    return vault.labels()


def _fit_point(distances: np.ndarray) -> tuple[float, float]:
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size < 3:
        return 1.0, float(np.mean(distances) if distances.size else 1.0)
    try:
        kappa, _loc, lam = weibull_min.fit(distances, floc=0)
    except Exception:
        return 1.0, float(np.mean(distances))
    if not np.isfinite(kappa) or not np.isfinite(lam) or lam <= 0 or kappa <= 0:
        return 1.0, float(np.mean(distances))
    return float(kappa), float(lam)


def fit(x_train: np.ndarray, y_train: np.ndarray, tau_evm: int) -> dict:
    y_train = np.asarray(y_train).astype(str)
    classes = np.unique(y_train)
    models = []
    failures = 0
    for label in classes:
        own = x_train[y_train == label]
        others = x_train[y_train != label]
        if others.shape[0] == 0 or own.shape[0] == 0:
            continue
        k = min(int(tau_evm), others.shape[0])
        neighbors = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(others)
        margins, _ = neighbors.kneighbors(own)
        for point, margin in zip(own, margins):
            kappa, lam = _fit_point(margin)
            if kappa == 1.0 and abs(lam - float(np.mean(margin))) < 1e-9:
                failures += 1
            models.append((label, point, kappa, lam))
    return {"models": models, "failures": failures, "tau_evm": int(tau_evm)}


def scores(model: dict, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_query = np.asarray(x_query, dtype=float)
    best = np.zeros(x_query.shape[0], dtype=float)
    pred = np.empty(x_query.shape[0], dtype=object)
    grouped: dict[str, list] = {}
    for label, point, kappa, lam in model["models"]:
        grouped.setdefault(label, []).append((point, kappa, lam))
    first = True
    for label, items in grouped.items():
        points = np.vstack([item[0] for item in items])
        distance = np.linalg.norm(x_query[:, None, :] - points[None, :, :], axis=2)
        psi = np.zeros_like(distance)
        for index, (_point, kappa, lam) in enumerate(items):
            psi[:, index] = np.exp(-((distance[:, index] / lam) ** kappa))
        class_score = psi.max(axis=1)
        take = first | (class_score > best)
        pred[take] = label
        best = np.maximum(best, class_score)
        first = False
    return 1.0 - best, pred
