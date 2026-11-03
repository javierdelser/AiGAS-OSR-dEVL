"""Oracle thresholds. This module is allowed to read test labels from the vault.

Every result produced here is an upper bound that uses unknown-class labels on
the test set. Callers must keep ``ORACLE`` in file names and captions.
"""

from __future__ import annotations

import numpy as np

from osr_static.metrics import apply_threshold, micro_f1_reference, micro_f1_sweep


def oracle_tau(vault, scores: np.ndarray, pred_closed) -> dict:
    """Maximum paper micro-F1 over thresholds of one score on the test set."""
    y_true = vault.labels()
    known_mask = vault.known_mask()
    known = [str(v) for v in vault._known.tolist()]
    f1s, taus = micro_f1_sweep(scores, pred_closed, y_true, known_mask)
    best = int(np.argmax(f1s))
    return {
        "f1": float(f1s[best]),
        "f1_percent": float(100.0 * f1s[best]),
        "tau": float(taus[best]) if np.isfinite(taus[best]) else None,
        "f1_curve": f1s,
        "tau_curve": taus,
        "curve_check": float(micro_f1_reference(y_true, apply_threshold(scores, pred_closed, taus[best]), known)),
    }


def oracle_q(vault, scores_by_q: dict[float, np.ndarray], pred_closed, tau: float = 1.0) -> dict:
    """Maximum test micro-F1 over a q grid at a fixed tau. Uses test labels."""
    y_true = vault.labels()
    known = [str(v) for v in vault._known.tolist()]
    best_q = None
    best_f1 = -1.0
    table = {}
    for q, scores in scores_by_q.items():
        guessed = apply_threshold(scores, pred_closed, tau)
        f1 = micro_f1_reference(y_true, guessed, known)
        table[f"{q:.3f}"] = float(100.0 * f1)
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and (best_q is None or abs(q - 0.85) < abs(best_q - 0.85))):
            best_f1 = f1
            best_q = q
    return {"q": best_q, "f1_percent": float(100.0 * best_f1), "by_q": table}
