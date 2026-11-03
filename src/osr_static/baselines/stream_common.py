"""Shared helpers for stream novelty detectors in the static OSR benchmark."""

from __future__ import annotations

import numpy as np

from osr_static.metrics import REJECT_LABEL, normalize_label

NOVEL_LABELS = frozenset({"NC", "NewClass"})


def map_novelty_predictions(pred: np.ndarray) -> np.ndarray:
    """Map paper novelty labels to the survey reject label ``UUC``."""
    out = np.asarray([str(p) for p in np.asarray(pred).ravel()], dtype=object)
    for label in NOVEL_LABELS:
        out[out == label] = normalize_label(REJECT_LABEL)
    return out
