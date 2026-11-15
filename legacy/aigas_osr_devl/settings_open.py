"""Output directory for the stream AiGAS driver (overridden by ``run_aigas_stream.py``)."""

from __future__ import annotations

import os
from pathlib import Path

path = os.environ.get("AIGAS_OUTPUT_DIR", "/tmp/aigas_osr_devl/")


def init() -> None:
    global path
    path = os.environ.get("AIGAS_OUTPUT_DIR", path)
    if not path.endswith("/"):
        path = path + "/"
    Path(path).mkdir(parents=True, exist_ok=True)