#!/usr/bin/env python3
"""Dynamic EVL stream benchmark (Table I) → ``results/osr_dynamic/SYNTHETIC_STREAM_TABLE.md``.

Prequential evaluation for MINAS, KNNENS, and SENCForest. Protocol constants and metrics
are in ``src/osr_static/stream_benchmark.py``, ``stream_eval.py``, and ``synthetic_metrics.py``
(same rules as the paper / legacy EVL driver).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from osr_static.paths import DATA_OSR_DYNAMIC, REPO_ROOT, RESULTS_DYNAMIC  # noqa: E402
from osr_static.stream_benchmark import TABLE_I  # noqa: E402
from osr_static.stream_eval import evaluate_stream_paper  # noqa: E402
from osr_static.synthetic import list_datasets  # noqa: E402

OUT_MD = RESULTS_DYNAMIC / "SYNTHETIC_STREAM_TABLE.md"


def _check_table_i_coverage() -> None:
    stems = {Path(name).stem for name in list_datasets()}
    missing_table = sorted(stems - set(TABLE_I))
    missing_data = sorted(set(TABLE_I) - stems)
    if missing_table:
        raise SystemExit(f"Datasets without Table I entry: {missing_table}")
    if missing_data:
        print(f"Note: Table I lists {missing_data} but no CSV under {DATA_OSR_DYNAMIC}", flush=True)


def _write_table(rows: list[dict]) -> None:
    by_ds: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], {})[row["method"]] = row
    lines = [
        "# Synthetic stream baselines (MINAS, KNNENS, SENCForest)",
        "",
        "Prequential stream protocol (Table I, arXiv:2504.05761): file order,",
        "`T_s` supervised prefix, batch size `n // batches`, sliding train window",
        "(`GNG_UTILS.definir_sets_training_test`). Labels train only for `t < T_s`;",
        "later batches update MINAS from unlabeled features (extra micro-clusters).",
        "",
        "Metrics: OSR micro-F1 (%) over all scored test instances; prequential error (%)",
        "= `errorVERDADERA` × 100 accumulated over stream batches (reject → 100).",
        "",
        "| Dataset | T_s | MINAS F1 | MINAS Preq | KNNENS F1 | KNNENS Preq | SENC F1 | SENC Preq |",
        "|---------|-----|----------|------------|-----------|-------------|---------|-----------|",
    ]
    for dataset in sorted(by_ds.keys()):
        cells = by_ds[dataset]
        ts = cells[next(iter(cells))].get("ts", float("nan"))
        line = [dataset, str(int(ts))]
        for method in ("minas", "knnens", "sencforest"):
            r = cells.get(method, {})
            line.append(f"{r.get('paper_f1_percent', float('nan')):.2f}")
            line.append(f"{r.get('prequential_error_percent', float('nan')):.2f}")
        lines.append("| " + " | ".join(line) + " |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.chdir(REPO_ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    _check_table_i_coverage()
    methods = ("minas", "knnens", "sencforest")
    rows: list[dict] = []
    for name in list_datasets():
        for method in methods:
            row = evaluate_stream_paper(name, method)
            rows.append(row)
            print(
                f"{method:10s} {row['dataset']:22s} "
                f"F1={row['paper_f1_percent']:.2f} preq={row['prequential_error_percent']:.2f} "
                f"steps={row['n_stream_steps']}",
                flush=True,
            )
    _write_table(rows)
    print(f"Wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
