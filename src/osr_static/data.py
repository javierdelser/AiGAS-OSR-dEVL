"""Dataset loading copied from ``load_txt_any`` in the legacy script.

The copy exists so partition indices can be built without importing ``osr_utils``
(that module parses ``sys.argv`` at import time). ``legacy_check`` asserts that
this function and the legacy one return the same arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_txt_any(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a headerless table. Label is the first non-numeric end column.

    Returns ``X`` as float and ``y`` as strings. This is the same rule as
    ``osr_GNG_Train_VAl_Test_LOOP_Split26.load_txt_any``.
    """
    df = pd.read_csv(path, header=None, sep=None, engine="python")
    arr = df.values
    first, last = arr[:, 0], arr[:, -1]

    def is_numeric(col: np.ndarray) -> bool:
        try:
            col.astype(float)
            return True
        except (TypeError, ValueError):
            return False

    first_num = is_numeric(first)
    last_num = is_numeric(last)

    if not first_num and last_num:
        y = first.astype(str)
        x = arr[:, 1:].astype(float)
    elif first_num and not last_num:
        y = last.astype(str)
        x = arr[:, :-1].astype(float)
    elif not first_num and not last_num:
        y = first.astype(str)
        x = arr[:, 1:].astype(float)
    else:
        y = last.astype(str)
        x = arr[:, :-1].astype(float)
    return x, y


class LeakageError(RuntimeError):
    """Raised when test labels are read from a module that is not allowed to see them."""


class LabelVault:
    """Test labels. ``known_mask`` is public. Raw labels are limited to metrics and oracle."""

    def __init__(self, y_true, known_classes: list[str]):
        from osr_static.metrics import normalize_label

        self._y = np.array([normalize_label(v) for v in np.asarray(y_true).ravel()], dtype=object)
        self._known = np.array([normalize_label(v) for v in known_classes], dtype=object)

    def known_mask(self) -> np.ndarray:
        return np.isin(self._y, self._known)

    def labels(self) -> np.ndarray:
        import inspect

        allowed = ("osr_static.metrics", "osr_static.oracle")
        for info in inspect.stack()[1:]:
            module = inspect.getmodule(info.frame)
            name = getattr(module, "__name__", "")
            if name.startswith(allowed):
                return self._y.copy()
        raise LeakageError("test labels are only available to osr_static.metrics and osr_static.oracle")
