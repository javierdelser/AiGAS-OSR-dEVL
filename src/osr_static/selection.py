"""Validation selection. This module must not read test labels."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from osr_static.metrics import apply_threshold, micro_f1_reference, micro_f1_sweep


Q_GRID = (0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 1.00)


def access_test_labels(vault):
    """Used by the leakage test. A direct call raises ``LeakageError``."""
    return vault.labels()


def select_threshold(scores: np.ndarray, pred_closed: Sequence[object], y_val, known_classes: Sequence[object]) -> tuple[float, float]:
    """Tau that maximises paper micro-F1 on the open validation set.

    Ties keep the threshold closest to the median finite score, then the smaller one.
    """
    scores = np.asarray(scores, dtype=float)
    known_mask = np.isin(np.array([str(v) for v in np.asarray(y_val).ravel()], dtype=object), [str(k) for k in known_classes])
    f1s, taus = micro_f1_sweep(scores, pred_closed, y_val, known_mask)
    best = float(np.max(f1s))
    finite_scores = scores[np.isfinite(scores)]
    center = float(np.median(finite_scores)) if finite_scores.size else 0.0
    candidates = []
    for f1, tau in zip(f1s, taus):
        if abs(float(f1) - best) <= 1e-12 and np.isfinite(tau):
            candidates.append(float(tau))
    if not candidates:
        return 1.0, best
    chosen = sorted(candidates, key=lambda tau: (abs(tau - center), tau))[0]
    return chosen, best


def select_q(val_f1_by_q: dict[float, float]) -> float:
    """q with the highest validation micro-F1. Ties: closest to 0.85, then smaller q."""
    best = max(val_f1_by_q.values())
    tied = [q for q, f1 in val_f1_by_q.items() if abs(f1 - best) <= 1e-12]
    return sorted(tied, key=lambda q: (abs(q - 0.85), q))[0]


def f1_at_tau(scores, pred_closed, y_true, known_classes, tau: float) -> float:
    guessed = apply_threshold(scores, pred_closed, tau)
    return micro_f1_reference(y_true, guessed, known_classes)
