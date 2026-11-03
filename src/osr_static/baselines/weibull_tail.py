"""Weibull tail used by W-SVM, W-OSVM, and PI-SVM."""

from __future__ import annotations

import numpy as np
from scipy.stats import weibull_min


def fit_upper(values: np.ndarray, tail_frac: float = 0.10, min_tail: int = 10) -> tuple[float, float, float]:
    """Weibull on the largest ``tail_frac`` of ``values``. Returns ``k, lam, shift``."""
    return _fit(values, upper=True, tail_frac=tail_frac, min_tail=min_tail)


def fit_lower(values: np.ndarray, tail_frac: float = 0.10, min_tail: int = 10) -> tuple[float, float, float]:
    """Weibull on the smallest ``tail_frac`` of ``values``. Returns ``k, lam, shift``."""
    return _fit(values, upper=False, tail_frac=tail_frac, min_tail=min_tail)


def survival(values: np.ndarray, k: float, lam: float, shift: float, upper: bool) -> np.ndarray:
    """Probability of being more extreme than ``values``. Clipped to ``(0, 1)``."""
    values = np.asarray(values, dtype=float)
    if upper:
        z = np.clip(values - shift, 1e-12, None)
    else:
        z = np.clip(shift - values, 1e-12, None)
    cdf = weibull_min.cdf(z, c=max(float(k), 1e-6), loc=0.0, scale=max(float(lam), 1e-8))
    return np.clip(1.0 - cdf, 1e-6, 1.0 - 1e-6)


def _fit(values: np.ndarray, upper: bool, tail_frac: float, min_tail: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0, 1.0, 0.0
    ordered = np.sort(values)
    n = max(int(min_tail), int(np.ceil(values.size * tail_frac)))
    n = min(n, ordered.size)
    tail = ordered[-n:] if upper else ordered[:n]
    shift = float(tail[0] if upper else tail[-1])
    z = np.clip((tail - shift) if upper else (shift - tail), 1e-9, None)
    try:
        kappa, _loc, lam = weibull_min.fit(z, floc=0)
    except Exception:
        return 1.0, float(np.mean(z)), shift
    if not np.isfinite(kappa) or not np.isfinite(lam) or kappa <= 0 or lam <= 0:
        return 1.0, float(np.mean(z)), shift
    return float(kappa), float(lam), shift
