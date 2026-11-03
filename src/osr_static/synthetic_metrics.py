"""Metrics aligned with ``Aigas_OSR_Open.py`` / ``osr_GNG_Train_VAl_Test_LOOP_Split26.py``."""

from __future__ import annotations

import numpy as np

# Open-set unknown / reject code in the legacy GNG scripts (flag == 1).
UNKNOWN_NUMERIC = 100
_REJECT = frozenset({"UUC", "NC", "NEWCLASS"})


def numeric_test_labels(y_te, y_tr) -> np.ndarray:
    """``my_y_test``: UUC or unseen train label → 100, else class id."""
    train_vals = {int(float(x)) for x in np.asarray(y_tr).ravel() if str(x) != "UUC"}
    out: list[int] = []
    for t in np.asarray(y_te).ravel():
        ts = str(t)
        if ts == "UUC":
            out.append(UNKNOWN_NUMERIC)
            continue
        v = int(float(ts))
        out.append(v if v in train_vals else UNKNOWN_NUMERIC)
    return np.array(out, dtype=int)


def numeric_predictions(y_pred) -> np.ndarray:
    """Map classifier output to legacy integer preds (reject → 100)."""
    out: list[int] = []
    for p in np.asarray(y_pred).ravel():
        ps = str(p).strip()
        if ps.upper() in _REJECT:
            out.append(UNKNOWN_NUMERIC)
        else:
            out.append(int(float(ps)))
    return np.array(out, dtype=int)


def prequential_error_percent(y_te, y_pred, y_tr) -> float:
    """``errorVERDADERA`` × 100 on one test batch (prequential at ``contador=0``)."""
    yt = numeric_test_labels(y_te, y_tr)
    yp = numeric_predictions(y_pred)
    return float(100.0 * np.mean(yt != yp))
