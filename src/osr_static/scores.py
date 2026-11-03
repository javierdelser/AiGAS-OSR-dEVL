"""Scores for the live 1_NN reject rule.

The legacy predictor rejects when the mean distance to the 5 neighbor GNG
nodes exceeds a threshold, or the maximum class probability falls below
another threshold (openness > 0 only). Higher score means more likely unknown.
"""

from __future__ import annotations

import numpy as np


def score_1nn(
    dist_mean: np.ndarray,
    confidence: np.ndarray,
    dist_threshold: float,
    conf_threshold: float,
) -> np.ndarray:
    """Ratio score. ``s > 1`` matches the legacy inequalities when thresholds are positive.

    ``s = max(dist / dist_threshold, conf_threshold / confidence)``.
    A zero distance threshold or a zero confidence makes that term infinite
    when it would force a reject, and 0 when the corresponding inequality
    cannot fire.
    """
    dist_mean = np.asarray(dist_mean, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if dist_threshold > 0:
        s_dist = dist_mean / float(dist_threshold)
    else:
        s_dist = np.where(dist_mean > dist_threshold, np.inf, 0.0)
    s_conf = np.divide(
        float(conf_threshold),
        confidence,
        out=np.full(confidence.shape, np.inf if conf_threshold > 0 else 0.0, dtype=float),
        where=confidence > 0,
    )
    return np.maximum(s_dist, s_conf)


def reject_mask_1nn(
    dist_mean: np.ndarray,
    confidence: np.ndarray,
    dist_threshold: float,
    conf_threshold: float,
    openness: float,
) -> np.ndarray:
    """Exact legacy predicate: openness > 0 and (dist > thr or conf < thr)."""
    dist_mean = np.asarray(dist_mean, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if openness <= 0:
        return np.zeros(dist_mean.shape[0], dtype=bool)
    return (dist_mean > dist_threshold) | (confidence < conf_threshold)


def linear_quantile(values: np.ndarray, q: float) -> float:
    """``numpy.quantile`` linear method, including NumPy 1.21's ``interpolation`` argument."""
    values = np.asarray(values, dtype=float)
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="linear"))


def class_centroids(
    x: np.ndarray,
    y: np.ndarray,
    q: float,
    min_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """One centroid per class. Radius is the ``q`` quantile of distances to that centroid.

    Returns centroids, radii, labels, and the number of classes with fewer than
    ``min_samples`` rows. Those classes are kept and counted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(str)
    centroids = []
    radii = []
    labels = []
    small = 0
    for label in sorted(set(y.tolist())):
        points = x[y == label]
        if points.shape[0] < min_samples:
            small += 1
        if points.shape[0] == 0:
            continue
        center = points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        radius = linear_quantile(distances, q)
        centroids.append(center)
        radii.append(max(radius, 1e-12))
        labels.append(label)
    return (
        np.asarray(centroids, dtype=float),
        np.asarray(radii, dtype=float),
        np.asarray(labels, dtype=object),
        small,
    )


def radius_scores(
    x: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``s_all``, ``s_nn``, ``s_raw`` and the nearest-centroid label.

    ``k*`` is the centroid with the smallest Euclidean distance. Higher scores
    are more unknown.
    """
    x = np.asarray(x, dtype=float)
    distance = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
    nearest = np.argmin(distance, axis=1)
    rows = np.arange(x.shape[0])
    s_raw = distance.min(axis=1)
    s_all = (distance / radii[None, :]).min(axis=1)
    s_nn = distance[rows, nearest] / radii[nearest]
    return s_all, s_nn, s_raw, labels[nearest]
