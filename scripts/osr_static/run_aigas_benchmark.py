#!/usr/bin/env python3
"""Run deployed **AiGAS-dEVL-OSR** static rescoring → ``results/osr_static/AIGAS_STATIC_TABLE.md``.

Requires saved Phase-1 GNG artifacts (JSON + NPZ). Point ``AIGAS_PHASE1_DIR`` at
``.../phase1/raw/phase1`` or ``.../results/osr_static/phase1/raw/phase1``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from aigas_osr_devl.static.deployed import DEPLOYED_RULES, deployed_rule_for, paper_f1_from_row  # noqa: E402
from osr_static.gng_mahalanobis import score_one as score_maha  # noqa: E402
from osr_static.gng_variants import score_one as score_variants  # noqa: E402
from osr_static.improvements import score_one as score_improve  # noqa: E402
from osr_static.paths import REPO_ROOT, RESULTS_STATIC  # noqa: E402


def _phase1_jsons(root: Path) -> list[Path]:
    return sorted(root.glob("*/*/O*/*.json"))


def _score(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    rule_needed = deployed_rule_for(record["dataset"], float(record["nominal_openness"]))
    if rule_needed.startswith("maha_"):
        return score_maha(str(path))
    if rule_needed.startswith("knn"):
        return score_variants(str(path))
    return score_improve(str(path))


def _write_table(rows: list[dict], out: Path) -> None:
    by: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        rule = deployed_rule_for(row["dataset"], float(row["nominal_openness"]))
        by[(row["dataset"], round(float(row["nominal_openness"]), 2))].append(paper_f1_from_row(row, rule))
    lines = [
        "# AiGAS-dEVL-OSR (deployed static rules)",
        "",
        "Paper micro-F1 (%). Mean ± std over partition replicates.",
        "",
        "| Dataset | Openness | Micro-F1 | Rule |",
        "|---------|----------|----------|------|",
    ]
    for (dataset, openness) in sorted(by.keys()):
        vals = by[(dataset, openness)]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        fmt = f"{mean:.2f}" if std != std else f"{mean:.2f} ± {std:.2f}"
        rule = DEPLOYED_RULES[(dataset.upper(), openness)]
        lines.append(f"| {dataset} | {openness:.2f} | {fmt} | `{rule}` |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1", default=os.environ.get("AIGAS_PHASE1_DIR", ""))
    args = parser.parse_args()
    phase1 = Path(args.phase1) if args.phase1 else None
    if not phase1 or not phase1.is_dir():
        raise SystemExit(
            "Set AIGAS_PHASE1_DIR to Phase-1 GNG outputs (directory containing LETTER/.../1_NN__*.json)."
        )
    jsons = [p for p in _phase1_jsons(phase1) if p.name.startswith("1_NN")]
    if not jsons:
        raise SystemExit(f"No 1_NN JSON under {phase1}")
    os.chdir(REPO_ROOT)
    rows = []
    for path in jsons:
        print("score", path.relative_to(phase1), flush=True)
        rows.append(_score(path))
    out = RESULTS_STATIC / "AIGAS_STATIC_TABLE.md"
    _write_table(rows, out)
    print(f"Wrote {out} ({len(rows)} partitions)", flush=True)


if __name__ == "__main__":
    main()