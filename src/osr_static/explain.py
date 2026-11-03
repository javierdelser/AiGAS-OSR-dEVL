"""Post-hoc feature attributions for AiGAS-dEVL-OSR.

The reject score is a mean distance to GNG nodes. That distance is a sum over
coordinates, so each feature has an additive share of the known-versus-unknown
decision. Euclidean shares are squared residuals. Mahalanobis shares are the
terms ``(x_j - n_j) (P (x - n))_j`` of the quadratic form.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from osr_static.gng_mahalanobis import ClassMetrics, fit_precision
from osr_static.improvements import N_NEIGHBORS, knn_predict

LETTER_FEATURES = [
    "xbox (box x)",
    "ybox (box y)",
    "width",
    "height",
    "onpix (on pixels)",
    "xbar (mean x)",
    "ybar (mean y)",
    "x2bar (x variance)",
    "y2bar (y variance)",
    "xybar (xy correlation)",
    "x2ybr",
    "xy2br",
    "x-ege (left-right edges)",
    "xegvy",
    "y-ege (bottom-top edges)",
    "yedgex",
]
PENDIGITS_FEATURES = [
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8",
]


def feature_names(dataset: str) -> list[str]:
    return list(LETTER_FEATURES if dataset == "LETTER" else PENDIGITS_FEATURES)


def euclidean_shares(query: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Mean over neighbors of ``(x_j - n_j)^2 / ||x - n||^2``."""
    query = np.asarray(query, dtype=float)
    neighbors = np.asarray(neighbors, dtype=float)
    delta = neighbors - query
    squared = delta ** 2
    denom = squared.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return (squared / denom).mean(axis=0)


def mahalanobis_shares(query: np.ndarray, neighbors: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Mean signed share of the Mahalanobis quadratic form, then L1-normalized."""
    query = np.asarray(query, dtype=float)
    neighbors = np.asarray(neighbors, dtype=float)
    precision = np.asarray(precision, dtype=float)
    delta = neighbors - query
    projected = delta @ precision
    raw = delta * projected
    total = np.abs(raw).sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return (raw / total).mean(axis=0)


def explain_query(
    query: np.ndarray,
    nodes: np.ndarray,
    node_labels: np.ndarray,
    metric: ClassMetrics | None,
    k: int = N_NEIGHBORS,
    distance: str = "euclidean",
) -> dict[str, Any]:
    """Neighborhood, predicted class, and per-feature shares for one scaled query."""
    query = np.asarray(query, dtype=float).reshape(1, -1)
    dist, conf, pred = knn_predict(nodes, node_labels, query, k)
    predicted = int(pred[0])
    model = NearestNeighbors(n_neighbors=min(int(k), nodes.shape[0]), algorithm="ball_tree")
    model.fit(nodes)
    neigh_dist, index = model.kneighbors(query)
    neighbors = nodes[index[0]]
    neighbor_labels = np.asarray(node_labels)[index[0]]
    if distance == "mahalanobis" and metric is not None:
        subset = nodes[node_labels == predicted]
        if subset.shape[0] == 0:
            subset = neighbors
        k_use = min(int(k), subset.shape[0])
        class_model = NearestNeighbors(n_neighbors=k_use, algorithm="ball_tree")
        class_model.fit(subset)
        class_dist, class_index = class_model.kneighbors(query)
        class_neighbors = subset[class_index[0]]
        precision = fit_precision(subset)
        shares = mahalanobis_shares(query[0], class_neighbors, precision)
        score = float(metric.score_known_label(query, np.array([predicted]), k)[0])
        used = class_neighbors
        used_dist = class_dist[0]
    else:
        shares = euclidean_shares(query[0], neighbors)
        score = float(dist[0])
        used = neighbors
        used_dist = neigh_dist[0]
    order = np.argsort(-np.abs(shares))
    return {
        "predicted_label": predicted,
        "confidence": float(conf[0]),
        "score": score,
        "neighbor_distances": used_dist.astype(float).tolist(),
        "neighbor_labels": [int(v) for v in neighbor_labels],
        "shares": shares.astype(float).tolist(),
        "ranked_features": [int(i) for i in order],
        "neighbors": used.astype(float).tolist(),
        "query": query[0].astype(float).tolist(),
    }


def decision_text(score: float, tau: float | None, known: bool, label: str) -> str:
    if tau is None:
        return f"Closed set. Predicted class {label}."
    unknown = score > tau
    if known and not unknown:
        return f"Accepted as known class {label}."
    if known and unknown:
        return f"Rejected a known {label} (score {score:.3f} > {tau:.3f})."
    if unknown:
        return f"Rejected as unknown (score {score:.3f} > {tau:.3f}). Predicted neighbor class {label}."
    return f"Accepted an unknown as class {label} (score {score:.3f} ≤ {tau:.3f})."
