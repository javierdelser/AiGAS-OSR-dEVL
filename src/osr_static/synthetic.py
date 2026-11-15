"""Dynamic EVL stream CSVs under ``data/osr_dynamic/``.

Each file is CSV with feature columns and a ``CLASS`` label (1..C).
A header line (``F1,F2,...,CLASS``) is optional; files without it are read
from their first line, so no sample is lost.
Known classes are inferred from the filename prefix (``2C`` → classes 1–2,
``4*`` → 1–4, ``5*`` → 1–5). Splits follow ``partitions._split_indices`` with
seed 123, same as LETTER/PENDIGITS.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osr_static.partitions import _split_indices, materialize, openness


def synthetic_root() -> Path:
    from osr_static.paths import DATA_DYNAMIC_STREAMS

    return DATA_DYNAMIC_STREAMS


def _has_header(path: Path) -> bool:
    """True when the first line holds column names rather than numbers."""
    with open(path, encoding="utf-8") as fh:
        first = fh.readline().strip().split(",")
    try:
        [float(v) for v in first]
    except ValueError:
        return True
    return False


def load_synthetic(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = synthetic_root() / name
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, header=0 if _has_header(path) else None)
    y = df.iloc[:, -1].astype(float).astype(int).astype(str)
    x = df.iloc[:, :-1].astype(float).values
    known = infer_known_classes(path.stem, y)
    return x, y, known


def infer_known_classes(stem: str, y: np.ndarray) -> list[str]:
    """Map dataset name to known label strings (1-indexed class ids in the file)."""
    upper = stem.upper()
    if "_2C_" in upper or re.match(r"^[12]C", upper):
        k = 2
    elif upper.startswith("4"):
        k = 4
    elif upper.startswith("5"):
        k = 5
    else:
        labels = sorted(int(float(v)) for v in np.unique(y))
        k = max(1, len(labels) - 1)
    return [str(i) for i in range(1, k + 1)]


def build_record(name: str) -> dict[str, Any]:
    x, y, known = load_synthetic(name)
    block = _split_indices(x, y, known)
    block["partition_id"] = 1
    block["nominal_openness"] = float(round(100.0 * block["actual_openness"], 2))
    block["dataset"] = Path(name).stem
    return block


def materialize_synthetic(name: str) -> dict[str, Any]:
    x, y, known = load_synthetic(name)
    record = build_record(name)
    arrays = materialize(x, y, record)
    arrays["record"] = record
    arrays["openness_percent"] = float(100.0 * record["actual_openness"])
    return arrays


def list_datasets() -> list[str]:
    return sorted(p.name for p in synthetic_root().glob("*.txt"))
