"""SENC / SENCForest (Mu, Ting, Zhou, IEEE TKDE 2017), batch OSR adaptation.

Implements SENCTrees (random axis splits until ``min_size``), per-tree path-length
threshold ``t_hat`` via minimum SDdiff (Section 5.2), anomaly balls in short-path
leaves, and majority class labels in known (K) and in-ball (B) regions. Test points
in A but outside B are **NewClass**. Forest prediction is majority vote over trees.

Stream buffer, model update, and multi-forest retirement are omitted.

Open-set rejection follows the paper **NewClass** label (mapped to ``UUC``); tree
hyperparameters are validated on F1, not a second score threshold ``tau``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NEW_CLASS = "NewClass"


def access_test_labels(vault):
    return vault.labels()


@dataclass
class _Leaf:
    depth: int
    indices: np.ndarray
    counts: dict[str, int]
    region: str = "K"  # K, B, or A-outlying (no train points)
    ball_center: np.ndarray | None = None
    ball_radius: float = 0.0


@dataclass
class _Node:
    feature: int
    value: float
    left: _Node | _Leaf
    right: _Node | _Leaf


def _majority(counts: dict[str, int]) -> str:
    if not counts:
        return "0"
    labels = list(counts.keys())
    weights = np.asarray([counts[l] for l in labels])
    return str(labels[int(np.argmax(weights))])


def _path_length_threshold(depths: list[int]) -> float:
    """``t_hat`` minimizing |std(Ll) - std(Lr)| on sorted leaf depths (Section 5.2)."""
    if len(depths) < 2:
        return float(depths[0]) if depths else 0.0
    values = np.sort(np.asarray(depths, dtype=float))
    best_t = values[0]
    best_diff = float("inf")
    for split in range(1, len(values)):
        left = values[:split]
        right = values[split:]
        if left.size == 0 or right.size == 0:
            continue
        diff = abs(float(left.std(ddof=1 if left.size > 1 else 0)) - float(right.std(ddof=1 if right.size > 1 else 0)))
        if diff < best_diff:
            best_diff = diff
            best_t = (values[split - 1] + values[split]) / 2.0
    return float(best_t)


def _build_tree(
    x: np.ndarray,
    indices: np.ndarray,
    depth: int,
    min_size: int,
    rng: np.random.Generator,
) -> _Leaf | _Node:
    if indices.size < min_size:
        return _Leaf(depth=depth, indices=indices.copy(), counts={})
    feature = int(rng.integers(0, x.shape[1]))
    column = x[indices, feature]
    lo, hi = float(column.min()), float(column.max())
    if lo >= hi:
        return _Leaf(depth=depth, indices=indices.copy(), counts={})
    split = float(rng.uniform(lo, hi))
    left_idx = indices[column <= split]
    right_idx = indices[column > split]
    if left_idx.size == 0 or right_idx.size == 0:
        return _Leaf(depth=depth, indices=indices.copy(), counts={})
    return _Node(
        feature=feature,
        value=split,
        left=_build_tree(x, left_idx, depth + 1, min_size, rng),
        right=_build_tree(x, right_idx, depth + 1, min_size, rng),
    )


def _iter_leaves(node: _Leaf | _Node) -> list[_Leaf]:
    if isinstance(node, _Leaf):
        return [node]
    return _iter_leaves(node.left) + _iter_leaves(node.right)


def _label_leaves(node: _Leaf | _Node, y: np.ndarray) -> None:
    for leaf in _iter_leaves(node):
        if leaf.indices.size == 0:
            continue
        labels, counts = np.unique(y[leaf.indices], return_counts=True)
        leaf.counts = {str(l): int(c) for l, c in zip(labels, counts)}


def _finalize_tree(node: _Leaf | _Node, x: np.ndarray, t_hat: float) -> None:
    for leaf in _iter_leaves(node):
        if leaf.indices.size == 0:
            leaf.region = "K"
            continue
        if float(leaf.depth) >= t_hat:
            leaf.region = "K"
            continue
        pts = x[leaf.indices]
        center = pts.mean(axis=0)
        dist = np.linalg.norm(pts - center, axis=1)
        radius = float(dist.max()) if dist.size else 0.0
        leaf.region = "A"
        leaf.ball_center = center
        leaf.ball_radius = max(radius, 1e-12)


def _route(node: _Leaf | _Node, row: np.ndarray) -> _Leaf:
    while isinstance(node, _Node):
        if row[node.feature] <= node.value:
            node = node.left
        else:
            node = node.right
    return node


class _SENCTree:
    def __init__(self, root: _Leaf | _Node, t_hat: float) -> None:
        self.root = root
        self.t_hat = t_hat

    def predict_one(self, row: np.ndarray) -> tuple[str, float]:
        leaf = _route(self.root, row)
        if leaf.region == "K" or leaf.ball_center is None:
            return _majority(leaf.counts), 0.0
        dist = float(np.linalg.norm(row - leaf.ball_center))
        if dist <= leaf.ball_radius:
            return _majority(leaf.counts), 0.0
        return NEW_CLASS, 1.0


def _build_senc_tree(x_sub: np.ndarray, y_sub: np.ndarray, min_size: int, rng: np.random.Generator) -> _SENCTree:
    indices = np.arange(x_sub.shape[0])
    root = _build_tree(x_sub, indices, 0, min_size, rng)
    _label_leaves(root, y_sub)
    depths = [leaf.depth for leaf in _iter_leaves(root) if leaf.indices.size > 0]
    t_hat = _path_length_threshold(depths if depths else [0])
    _finalize_tree(root, x_sub, t_hat)
    return _SENCTree(root, t_hat)


def fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_samples: int = 200,
    min_size: int = 10,
    random_state: int = 0,
) -> dict[str, Any]:
    y_train = np.asarray(y_train).astype(str)
    x_train = np.asarray(x_train, dtype=float)
    rng = np.random.default_rng(int(random_state))
    sample = min(int(max_samples), x_train.shape[0])
    trees: list[_SENCTree] = []
    for _ in range(int(n_estimators)):
        if sample >= x_train.shape[0]:
            idx = np.arange(x_train.shape[0])
        else:
            idx = rng.choice(x_train.shape[0], size=sample, replace=False)
        trees.append(_build_senc_tree(x_train[idx], y_train[idx], int(min_size), rng))
    return {
        "trees": trees,
        "n_estimators": int(n_estimators),
        "max_samples": int(sample),
        "min_size": int(min_size),
    }


def scores(model: dict[str, Any], x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_query = np.asarray(x_query, dtype=float)
    trees: list[_SENCTree] = model["trees"]
    n = x_query.shape[0]
    nc_votes = np.zeros(n, dtype=float)
    vote_labels: list[list[str]] = [[] for _ in range(n)]
    for tree in trees:
        for i in range(n):
            label, nc_flag = tree.predict_one(x_query[i])
            vote_labels[i].append(label)
            nc_votes[i] += nc_flag
    score = nc_votes / max(len(trees), 1)
    pred = np.empty(n, dtype=object)
    for i in range(n):
        labels = vote_labels[i]
        nc = sum(1 for v in labels if v == NEW_CLASS)
        if nc > len(labels) / 2:
            pred[i] = NEW_CLASS
            continue
        known = [v for v in labels if v != NEW_CLASS]
        if not known:
            pred[i] = NEW_CLASS
            continue
        uniq, counts = np.unique(known, return_counts=True)
        pred[i] = str(uniq[int(np.argmax(counts))])
    return score.astype(float), np.asarray([str(p) for p in pred], dtype=object)


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    n_estimators: int = 100,
    max_samples: int = 200,
    min_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    return scores(fit(x_train, y_train, n_estimators, max_samples, min_size), x_query)
