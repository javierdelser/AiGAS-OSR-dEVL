"""Benchmark stream layout (Table I, 2504.05761v2 / EVL repository)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

# Paper Table I: instances / batches / T_s (supervised prefix length).
TABLE_I: dict[str, tuple[int, int]] = {
    "1CDT_MOD_OSR": (800, 100),
    "1CHT_MOD_OSR": (800, 100),
    "2CDT_MOD_OSR": (800, 100),
    "2CHT_MOD_OSR": (800, 100),
    "5CVT_MOD_OSR": (1000, 200),
    "1CSurr_MOD_OSR": (920, 300),
    "GEARS_2C_2D_SI_OSR": (910, 1095),
    "4CR_NO_OSR": (7220, 100),
    "4CRE-V1_NO_OSR": (1250, 500),
    "4CRE-V2_NO_OSR": (1140, 800),
    "UG_2C_2D_SI_OSR": (2500, 200),
    "UG_2C_3D_SI_OSR": (2000, 200),
    "UG_2C_5D_SI_OSR": (2000, 500),
    "4CE1CF_NO_OSR": (4330, 200),
    "FG_2C_2D": (5000, 200),
    "MG_2C_2D": (5000, 200),
}


@dataclass(frozen=True)
class StreamLayout:
    n: int
    ts: int
    vol_batch: int
    window: int
    n_steps: int


def layout_for(n: int, stem: str) -> StreamLayout:
    if stem not in TABLE_I:
        raise KeyError(f"No Table I entry for {stem}")
    ts, batches = TABLE_I[stem]
    vol = max(1, n // batches)
    ts = _align_ts(ts, vol, n)
    window = ts
    steps = max(0, (n - window) // vol)
    return StreamLayout(n=n, ts=ts, vol_batch=vol, window=window, n_steps=steps)


def _align_ts(ts: int, vol: int, n: int) -> int:
    """Same snapping as ``Aigas_OSR_Open.py`` (multiple of ``vol_batch``)."""
    if ts < vol:
        ts = vol
    elif ts % vol != 0:
        ts = (ts // vol) * vol
    return min(ts, n - vol)


def batch_slices(layout: StreamLayout) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """``definir_sets_training_test`` index ranges: train window then test chunk."""
    n, vol, win = layout.n, layout.vol_batch, layout.window
    for contador in range(layout.n_steps):
        train_start = contador * vol
        train_end = train_start + win
        test_start = train_end
        test_end = test_start + vol
        if test_end > n:
            break
        yield np.arange(train_start, train_end, dtype=int), np.arange(test_start, test_end, dtype=int)
