"""Phase 1 orchestration: partitions, legacy jobs, reproduction report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from osr_static.jobs import config_hash, derive_seed
from osr_static.paths import CONFIG_DIR, DATA_STATIC, REPO_ROOT, RESULTS_STATIC, SRC_DIR


def _exp_root() -> Path:
    return REPO_ROOT


def _repo_root() -> Path:
    return REPO_ROOT


def _load_config(path: Path) -> dict[str, Any]:
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


def _parse_only(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    out: dict[str, str] = {}
    for piece in text.split(","):
        key, value = piece.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _select(jobs: list[dict[str, Any]], only: dict[str, str]) -> list[dict[str, Any]]:
    selected = []
    for job in jobs:
        if "dataset" in only and job["dataset"].lower() != only["dataset"].lower():
            continue
        if "openness" in only and abs(float(job["nominal_openness"]) - float(only["openness"])) > 1e-9:
            continue
        if "partition" in only and int(job["partition_id"]) != int(only["partition"]):
            continue
        if "method" in only and job["method"].lower() != only["method"].lower():
            continue
        selected.append(job)
    return selected


def _build_partitions(config: dict[str, Any], garage: Path) -> None:
    exp = _exp_root()
    repo = _repo_root()
    request = {
        "repo_root": str(repo),
        "data_dir": str(DATA_STATIC),
        "out_dir": str(DATA_STATIC / "partitions"),
        "datasets": config["datasets"],
    }
    path = DATA_STATIC / "partitions" / "_build_request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [str(garage), "-m", "osr_static.build_cli", "--request", str(path)],
        cwd=str(repo),
        env=env,
        check=True,
    )


def _jobs(config: dict[str, Any], epochs: int, tag: str) -> list[dict[str, Any]]:
    exp = _exp_root()
    repo = _repo_root()
    hashed = config_hash(
        {
            "method": "1_NN",
            "epochs": epochs,
            "datasets": config["datasets"],
            "split_seed": 123,
        }
    )
    jobs = []
    for dataset, spec in config["datasets"].items():
        document = json.loads((DATA_STATIC / "partitions" / f"{dataset}.json").read_text(encoding="utf-8"))
        for record in document["partitions"]:
            nominal = float(record["nominal_openness"])
            partition_id = int(record["partition_id"])
            seed = derive_seed(dataset, partition_id, nominal, "1_NN", hashed)
            nominal_name = f"{nominal:.2f}"
            out_dir = (
                RESULTS_STATIC / tag / "raw" / "phase1" / dataset
                / f"p{partition_id}" / f"O{nominal_name}"
            )
            stem = out_dir / f"1_NN__{hashed}"
            jobs.append(
                {
                    "dataset": dataset,
                    "data_file": spec["file"],
                    "partition_id": partition_id,
                    "nominal_openness": nominal,
                    "method": "1_NN",
                    "epochs": epochs,
                    "seed": seed,
                    "config_hash": hashed,
                    "repo_root": str(repo),
                    "partition_file": str(DATA_STATIC / "partitions" / f"{dataset}.json"),
                    "out_json": str(stem.with_suffix(".json")),
                    "out_npz": str(stem.with_suffix(".npz")),
                    "failed_json": str(RESULTS_STATIC / tag / "failed" / f"{dataset}_p{partition_id}_O{nominal_name}.json"),
                    "auroc_prefix": str(out_dir / "auroc_"),
                }
            )
    return jobs


def _run_one(garage: Path, job: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    out_json = Path(job["out_json"])
    if out_json.exists():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
        if existing.get("ok") and int(existing.get("decision_mismatches", 1)) == 0:
            print(f"resume {job['dataset']} p{job['partition_id']} O{job['nominal_openness']}", flush=True)
            return existing
    exp = _exp_root()
    spec_path = out_json.with_suffix(".job.json")
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(job), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    log_path = log_dir / f"{job['dataset']}_p{job['partition_id']}_O{float(job['nominal_openness']):.2f}.log"
    print(f"start {log_path.name}", flush=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [str(garage), "-m", "osr_static.worker", "--job", str(spec_path)],
            cwd=str(_repo_root()),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0 or not out_json.exists():
        return {
            "ok": False,
            "dataset": job["dataset"],
            "partition_id": job["partition_id"],
            "nominal_openness": job["nominal_openness"],
            "returncode": completed.returncode,
            "log": str(log_path),
        }
    return json.loads(out_json.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1))


def _fmt(mean: float, std: float) -> str:
    if std != std:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def write_report(
    records: list[dict[str, Any]], paper_path: Path | None, out_path: Path, jobs_parallel: int
) -> None:
    import csv

    paper = {}
    if paper_path is not None and paper_path.is_file():
        with open(paper_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                paper[(row["dataset"], round(float(row["openness_percent"]), 2))] = (
                    float(row["micro_f1_mean"]),
                    float(row["micro_f1_std"]),
                )
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for record in records:
        if not record.get("ok"):
            continue
        key = (record["dataset"], round(float(record["nominal_openness"]), 2))
        groups.setdefault(key, []).append(record)

    lines = [
        "# Phase 1 reproduction",
        "",
        "Proposed method: legacy GNG + `1_NN`, partitions taken from `data/osr_static/*.log`.",
        "Two micro-F1 numbers are reported. **Repo** is `OSRMetrics.micro_f1` (unknowns predicted as a known class are the only false positives). **Paper** counts every wrong assignment to a known class, including confusions between known classes.",
        "A cell passes when `|mean − paper| ≤ max(paper std, 0.5)`. The tolerance uses the standard deviation printed in the paper.",
        "",
        "| Dataset | Nominal O | Actual O | n | Paper | Repo F1 | Paper F1 | Repo | Paper-def | FRR | FAR | closed acc | mismatches |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    any_fail = False
    for key in sorted(groups, key=lambda item: (item[0], item[1])):
        rows = groups[key]
        paper_mean, paper_std = paper.get(key, (float("nan"), float("nan")))
        repo_mean, repo_std = _mean_std([r["repo_f1_percent"] for r in rows])
        paper_def_mean, paper_def_std = _mean_std([r["paper_f1_percent"] for r in rows])
        frr, _ = _mean_std([r["decomposition"]["FRR"] for r in rows])
        far, _ = _mean_std([r["decomposition"]["FAR"] for r in rows])
        closed, _ = _mean_std([r["decomposition"]["closed_acc"] for r in rows])
        mismatches = sum(int(r["decision_mismatches"]) for r in rows)
        score_mismatches = sum(int(r["score_gt1_mismatches"]) for r in rows)
        actual = rows[0]["actual_openness_percent"]
        tol = max(paper_std, 0.5) if paper_std == paper_std else 0.5
        repo_pass = abs(repo_mean - paper_mean) <= tol
        paper_pass = abs(paper_def_mean - paper_mean) <= tol
        any_fail = any_fail or not (repo_pass and paper_pass)
        lines.append(
            "| {ds} | {nom:.2f} | {act:.2f} | {n} | {paper} | {repo} | {pdef} | {rp} | {pp} | {frr:.3f} | {far:.3f} | {cl:.3f} | {mm} / score {sm} |".format(
                ds=key[0],
                nom=key[1],
                act=actual,
                n=len(rows),
                paper=_fmt(paper_mean, paper_std),
                repo=_fmt(repo_mean, repo_std),
                pdef=_fmt(paper_def_mean, paper_def_std),
                rp="pass" if repo_pass else "fail",
                pp="pass" if paper_pass else "fail",
                frr=frr,
                far=far,
                cl=closed,
                mm=mismatches,
                sm=score_mismatches,
            )
        )

    failed = [r for r in records if not r.get("ok")]
    times = [float(r["train_seconds"]) for r in records if r.get("ok")]
    mean_t, std_t = _mean_std(times)
    total = sum(times)
    lines.extend(
        [
            "",
            "## Decision check",
            "",
            "For every test sample the legacy `1_NN` label was compared with the label obtained from the same nodes by the predicate `openness > 0 and (mean distance > threshold or confidence < threshold)`. `score > 1` uses `max(distance/threshold, confidence_threshold/confidence)`.",
            "",
            f"Jobs with a nonzero legacy-predicate mismatch: {sum(1 for r in records if r.get('ok') and r['decision_mismatches'])}.",
            f"Sum of predicate mismatches: {sum(int(r.get('decision_mismatches', 0)) for r in records if r.get('ok'))}.",
            f"Sum of `s > 1` mismatches: {sum(int(r.get('score_gt1_mismatches', 0)) for r in records if r.get('ok'))}.",
            "",
            "## Timing",
            "",
            f"Successful jobs: {len(times)}. Sum of train times {total/3600:.2f} h. Mean per job {_fmt(mean_t, std_t)} s.",
            f"Parallel workers requested: {jobs_parallel}. Wall time is about the sum divided by that width, plus interpreter startup.",
            "",
            "E1 and E3, if they rescore the saved distances and confidences, do not train another GNG. "
            "E2's radius grid does not apply to `1_NN`: the reject parameters are the two validation thresholds already stored. "
            "A rerun of this same 18-job grid costs the measured train time. "
            "Openness above 0 trains the GNG twice inside `run_method` (validation fit, then all known classes); openness 0 trains it once. That cost is included in the times above.",
            "",
            "## Why a comparison to Table 1 can fail",
            "",
            "- The paper's operating point is an 85th-percentile radius. This run is the repository's `1_NN` rule.",
            "- Nominal openness 8.71 is 8 known Pendigits classes (actual openness 5.72). Nominal 15.48 is 18 known Letter classes (actual openness 9.55). Only 18.35 and 25.46 match the survey's known-class counts.",
            "- Closed-set logs repeat one class list. The sample split is the same (seed 123); the GNG shuffle seed includes `partition_id`, so the two lines are not identical networks.",
            "- Fitting classes are the first `int(2/3·|KKC|)` labels in the log, which is 6 rather than 7 when there are 10 known classes.",
            "- The repository micro-F1 ignores known-versus-known false positives. The paper definition does not.",
            "",
        ]
    )
    if failed:
        lines.append("## Failed jobs")
        lines.append("")
        for record in failed:
            lines.append(
                f"- {record.get('dataset')} partition {record.get('partition_id')} "
                f"O {record.get('nominal_openness')}: {record.get('log', record.get('error', 'failed'))}"
            )
        lines.append("")
    lines.append(f"Report written at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_repro.yaml")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--garage", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    exp = _exp_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = _load_config(config_path)
    experiment = str(config.get("experiment", "phase1_repro"))
    only = _parse_only(args.only)
    if experiment in {"e1", "e2", "e3", "diagnostics"}:
        from osr_static.diagnose import run_diagnostics

        print(json.dumps(run_diagnostics(config, only, args.jobs)), flush=True)
        return
    if experiment in {"baselines", "smoke"}:
        from osr_static.baseline_runner import run_baselines

        tag = args.tag or ("smoke" if experiment == "smoke" else "baselines")
        print(json.dumps(run_baselines(config, only, args.jobs, tag=tag)), flush=True)
        return
    if experiment == "aggregate":
        from osr_static.aggregate import run_aggregate

        print(json.dumps(run_aggregate()), flush=True)
        return
    if experiment == "improvements":
        from osr_static.improvements import run_improvements

        print(json.dumps(run_improvements(only, jobs=args.jobs, resume=args.resume)), flush=True)
        return
    if experiment == "gng_variants":
        from osr_static.gng_variants import run_gng_variants

        print(json.dumps(run_gng_variants(only, jobs=args.jobs, resume=args.resume)), flush=True)
        return
    if experiment == "gng_options":
        from osr_static.gng_options import run_gng_options

        print(json.dumps(run_gng_options(only, jobs=args.jobs, resume=args.resume)), flush=True)
        return
    if experiment == "gng_mahalanobis":
        from osr_static.gng_mahalanobis import run_gng_mahalanobis

        print(json.dumps(run_gng_mahalanobis(only, jobs=args.jobs, resume=args.resume)), flush=True)
        return
    if experiment == "holdout":
        from osr_static.holdout_train import run_holdout

        garage = Path(args.garage) if args.garage else _repo_root() / "garage" / "bin" / "python"
        print(
            json.dumps(
                run_holdout(
                    epochs=int(args.epochs if args.epochs is not None else config.get("epochs", 50)),
                    jobs=args.jobs,
                    include_closed=bool(config.get("include_closed", False)),
                    garage=garage,
                )
            ),
            flush=True,
        )
        return
    _main_phase1(args, config)


def _main_phase1(args, config) -> None:
    exp = _exp_root()
    epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 50))
    garage = Path(args.garage) if args.garage else _repo_root() / "garage" / "bin" / "python"
    if not garage.exists():
        raise SystemExit(f"garage python not found at {garage}")
    tag = args.tag or "phase1"
    _build_partitions(config, garage)
    jobs = _select(_jobs(config, epochs, tag), _parse_only(args.only))
    if not jobs:
        raise SystemExit("no jobs selected")
    log_dir = RESULTS_STATIC / tag / "logs" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs, {args.jobs} workers, epochs={epochs}", flush=True)
    started = time.time()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(_run_one, garage, job, log_dir) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if record.get("ok"):
                print(
                    f"done {record['dataset']} p{record['partition_id']} "
                    f"O{record['nominal_openness']} repoF1={record['repo_f1_percent']:.2f} "
                    f"paperF1={record['paper_f1_percent']:.2f} mismatches={record['decision_mismatches']} "
                    f"in {record['train_seconds']:.0f}s",
                    flush=True,
                )
            else:
                print(f"FAILED {record}", flush=True)
    records.sort(key=lambda r: (str(r.get("dataset")), float(r.get("nominal_openness", 0)), int(r.get("partition_id", 0))))
    report = RESULTS_STATIC / tag / "phase1_reproduction.md"
    paper_csv = REPO_ROOT / "reference" / "table1_paper.csv"
    write_report(records, paper_csv if paper_csv.is_file() else None, report, args.jobs)
    summary = {
        "ok_jobs": sum(1 for r in records if r.get("ok")),
        "failed_jobs": sum(1 for r in records if not r.get("ok")),
        "wall_seconds": time.time() - started,
        "report": str(report),
    }
    (RESULTS_STATIC / tag / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if summary["failed_jobs"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
