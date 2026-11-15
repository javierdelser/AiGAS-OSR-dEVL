#!/usr/bin/env python3
"""Run AiGAS-OSR-dEVL on EVL streams via ``legacy/aigas_osr_devl/``.

Needs Python 3.8 and NeuPy (local ``garage/`` env or ``AIGAS_PYTHON``). Logs and
figures are written under ``AIGAS_WORK`` (default ``.aigas_work/``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DRIVER = _REPO / "legacy" / "aigas_osr_devl" / "run_aigas_stream.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not DRIVER.is_file():
        raise SystemExit(f"Missing driver {DRIVER}")
    py = os.environ.get("AIGAS_PYTHON")
    if not py:
        garage = _REPO / "garage" / "bin" / "python"
        py = str(garage) if garage.is_file() else sys.executable
    if shutil.which(py) is None and not Path(py).is_file():
        raise SystemExit(
            f"Interpreter {py!r} not found. Set AIGAS_PYTHON to a Python 3.8 env with NeuPy."
        )
    print(f"Running AiGAS stream driver with {py}", flush=True)
    subprocess.run([py, str(DRIVER)], check=True)


if __name__ == "__main__":
    main()