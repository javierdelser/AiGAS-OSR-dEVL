#!/usr/bin/env python3
"""Stream driver entry point: ``Aigas_OSR_Open.py`` on ``data/osr_dynamic/*.txt``.

NeuPy (Python 3.8) and ``requirements.txt``. Work directory: ``AIGAS_WORK``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parent
REPO = LEGACY.parents[1]
DATA = REPO / "data" / "osr_dynamic"
WORK = Path(os.environ.get("AIGAS_WORK", REPO / ".aigas_work"))

os.environ.setdefault("AIGAS_DATA_DIR", str(DATA))
os.environ.setdefault("AIGAS_OUTPUT_DIR", str(WORK / "streams"))
os.environ["MPLBACKEND"] = "Agg"
os.chdir(LEGACY)
sys.path.insert(0, str(LEGACY))

if __name__ == "__main__":
    import runpy

    print(f"Data: {DATA}", flush=True)
    print(f"Work: {WORK}", flush=True)
    runpy.run_path(str(LEGACY / "Aigas_OSR_Open.py"), run_name="__main__")