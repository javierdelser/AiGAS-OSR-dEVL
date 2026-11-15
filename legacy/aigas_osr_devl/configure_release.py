"""Dataset list for ``Aigas_OSR_Open.py`` against ``data/osr_dynamic/*.txt``."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Table I batch counts (paper); used like legacy ``dim_TOTAL``.
TABLE_I_BATCHES: dict[str, int] = {
    "1CDT_MOD_OSR": 100,
    "1CHT_MOD_OSR": 100,
    "2CDT_MOD_OSR": 100,
    "2CHT_MOD_OSR": 100,
    "5CVT_MOD_OSR": 200,
    "1CSurr_MOD_OSR": 300,
    "GEARS_2C_2D_SI_OSR": 1095,
    "4CR_NO_OSR": 100,
    "4CRE-V1_NO_OSR": 500,
    "4CRE-V2_NO_OSR": 800,
    "UG_2C_2D_SI_OSR": 200,
    "UG_2C_3D_SI_OSR": 200,
    "UG_2C_5D_SI_OSR": 500,
    "4CE1CF_NO_OSR": 200,
}


def data_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    override = os.environ.get("AIGAS_DATA_DIR")
    if override:
        return Path(override)
    return root / "data" / "osr_dynamic"


def stream_config() -> tuple[list[str], list[int], list[int]]:
    files = sorted(data_dir().glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No stream CSV under {data_dir()}")
    file_names: list[str] = []
    dim_total: list[int] = []
    num_features: list[int] = []
    for path in files:
        stem = path.stem
        df = pd.read_csv(path, nrows=2)
        file_names.append(path.name)
        num_features.append(int(df.shape[1] - 1))
        dim_total.append(TABLE_I_BATCHES.get(stem, max(1, len(df) // 5)))
    return file_names, dim_total, num_features