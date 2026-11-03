#!/usr/bin/env python3
"""Static OSR stream benchmark (LETTER / PENDIGITS) → ``results/osr_static/STREAM_BASELINES.md``.

Runs MINAS, KNNENS, and SENCForest on the shared partitions, aggregates micro-F1,
and writes a single markdown table. Job JSON is kept in a temporary work directory only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from osr_static.baseline_runner import run_baselines  # noqa: E402
from osr_static.paths import REPO_ROOT, RESULTS_STATIC  # noqa: E402


def _load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not a mapping")
    extends = payload.pop("extends", None)
    if not extends:
        return payload
    parent = path.parent / str(extends)
    base = _load_config(parent)
    merged = dict(base)
    merged.update(payload)
    return merged


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1))


def _fmt(mean: float, std: float) -> str:
    if mean != mean:
        return "—"
    if std != std:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def _collect_rows(work: Path) -> list[dict]:
    raw = work / "stream" / "raw"
    rows: list[dict] = []
    for path in sorted(raw.glob("*/*/*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("ok"):
            rows.append(payload)
    return rows


def _write_table(rows: list[dict], out: Path) -> None:
    grouped: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], round(float(row["nominal_openness"]), 2), row["method"])
        grouped[key].append(float(row["paper_f1_percent"]))
    methods = ["minas", "knnens", "sencforest"]
    settings = [
        ("LETTER", 0.0),
        ("LETTER", 15.48),
        ("LETTER", 25.46),
        ("PENDIGITS", 0.0),
        ("PENDIGITS", 8.71),
        ("PENDIGITS", 18.35),
    ]
    lines = [
        "# Stream-style baselines (MINAS, KNNENS, SENCForest)",
        "",
        "Paper micro-F1 (%). Open cells: mean ± std over partition replicates.",
        "",
        "| Dataset | Openness | MINAS | KNNENS | SENCForest |",
        "|---------|----------|-------|--------|------------|",
    ]
    for dataset, openness in settings:
        cells = [dataset, f"{openness:.2f}"]
        for method in methods:
            vals = grouped.get((dataset, round(openness, 2), method), [])
            mean, std = _mean_std(vals)
            cells.append(_fmt(mean, std))
        lines.append("| " + " | ".join(cells) + " |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--config", default="configs/baselines_stream.yaml")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    config_path = REPO_ROOT / args.config
    config = _load_config(config_path)
    out_md = RESULTS_STATIC / "STREAM_BASELINES.md"
    with tempfile.TemporaryDirectory(prefix="osr_static_bench_") as tmp:
        os.environ["OSR_BENCHMARK_WORK"] = tmp
        print(f"Running stream baselines with {args.cpus} workers (scratch: {tmp})", flush=True)
        run_baselines(config, only={}, n_jobs=max(1, args.cpus), tag="stream")
        rows = _collect_rows(Path(tmp))
        if not rows:
            raise SystemExit("No successful jobs; table not written.")
        _write_table(rows, out_md)
    print(f"Wrote {out_md} ({len(rows)} jobs)", flush=True)


if __name__ == "__main__":
    main()
