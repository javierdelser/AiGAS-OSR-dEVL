"""Data, script, and results paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_STATIC = REPO_ROOT / "data" / "osr_static"
DATA_OSR_DYNAMIC = REPO_ROOT / "data" / "osr_dynamic"
# Back-compat alias
DATA_DYNAMIC_STREAMS = DATA_OSR_DYNAMIC
SCRIPTS_DYNAMIC = REPO_ROOT / "scripts" / "osr_dynamic"

RESULTS_STATIC = REPO_ROOT / "results" / "osr_static"
RESULTS_DYNAMIC = REPO_ROOT / "results" / "osr_dynamic"

CONFIG_DIR = REPO_ROOT / "configs"
SCRIPTS_STATIC = REPO_ROOT / "scripts" / "osr_static"
SRC_DIR = REPO_ROOT / "src"


def benchmark_work_root() -> Path:
    """Writable directory for per-job JSON during benchmark runs."""
    import os

    override = os.environ.get("OSR_BENCHMARK_WORK")
    if override:
        return Path(override)
    return RESULTS_STATIC
