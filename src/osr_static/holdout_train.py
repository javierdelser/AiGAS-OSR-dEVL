"""Train one GNG per open partition on a row holdout, then cut distance on the held-out rows.

The saved networks were grown on every known-class training row, so a distance
cut measured on those rows is in-sample and the simulated unknowns sit inside
the graph. This run grows the network on 80% of those rows. The other 20% of
every known class never enters the graph. The distance cut is the 0.95 quantile
of that holdout. Openness 0 is skipped unless ``--include-closed`` is set:
rejection is off there, and those two Letter jobs are the long ones.

Requires the garage interpreter (NeuPy / Theano). The baseline virtualenv cannot
train a GNG.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from osr_static.jobs import atomic_json, config_hash, dataset_file, derive_seed, partition_file
from osr_static.paths import REPO_ROOT, RESULTS_STATIC, SRC_DIR
from osr_static.paths import RESULTS_STATIC
from osr_static.partitions import load_partitions

DATA_FILES = {
    "LETTER": "letter_dataset.csv",
    "PENDIGITS": "pendigits_dataset.csv",
}
HOLDOUT_FRACTION = 0.20
COVERAGE = 0.95


def holdout_indices(y: Sequence[object], fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-class row split. Calibration rows are disjoint from the grow rows and cover no extra rows."""
    y = np.asarray(y).astype(str)
    rng = np.random.default_rng(int(seed))
    grow_parts = []
    cal_parts = []
    for label in sorted(set(y.tolist())):
        local = np.flatnonzero(y == label)
        local = rng.permutation(local)
        n_cal = int(round(float(fraction) * local.size))
        n_cal = min(max(n_cal, 1), local.size - 1)
        cal_parts.append(local[:n_cal])
        grow_parts.append(local[n_cal:])
    return np.concatenate(grow_parts), np.concatenate(cal_parts)


def _repo_root() -> Path:
    return REPO_ROOT


def _jobs(epochs: int, include_closed: bool, fraction: float) -> list[dict[str, Any]]:
    hashed = config_hash(
        {
            "method": "holdout_1nn",
            "epochs": int(epochs),
            "holdout_fraction": float(fraction),
            "coverage": COVERAGE,
            "split_seed": 123,
        }
    )
    jobs = []
    for dataset, filename in DATA_FILES.items():
        document = load_partitions(partition_file(dataset))
        for record in document["partitions"]:
            nominal = float(record["nominal_openness"])
            if not include_closed and float(record["actual_openness"]) <= 0:
                continue
            partition_id = int(record["partition_id"])
            nominal_name = f"{nominal:.2f}"
            out_dir = RESULTS_STATIC / "holdout" / "raw" / dataset / f"p{partition_id}" / f"O{nominal_name}"
            stem = out_dir / f"holdout__{hashed}"
            jobs.append(
                {
                    "dataset": dataset,
                    "data_file": filename,
                    "partition_file": str(partition_file(dataset)),
                    "partition_id": partition_id,
                    "nominal_openness": nominal,
                    "epochs": int(epochs),
                    "holdout_fraction": float(fraction),
                    "seed": derive_seed(dataset, partition_id, nominal, "holdout_1nn", hashed),
                    "config_hash": hashed,
                    "repo_root": str(_repo_root()),
                    "out_json": str(stem.with_suffix(".json")),
                    "out_npz": str(stem.with_suffix(".npz")),
                    "failed_json": str(out_dir / "failed.json"),
                }
            )
    return jobs


def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    """Train one holdout network. Call this only from the garage interpreter."""
    out_json = Path(job["out_json"])
    if out_json.is_file():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
        if existing.get("ok"):
            return existing
    try:
        record = _train(job)
    except Exception as exc:
        failed = {"ok": False, "error": str(exc), "traceback": traceback.format_exc(), "job": job}
        atomic_json(job["failed_json"], failed)
        raise
    atomic_json(out_json, record)
    return record


def _train(job: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(job["repo_root"])
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    sys.argv = [sys.argv[0]]

    import osr_GNG_Train_VAl_Test_LOOP_Split26 as loop
    from osr_static.data import load_txt_any
    from osr_static.improvements import closed_labels, knn_predict, relabel_from_nearest, to_numeric
    from osr_static.metrics import decompose, micro_f1_reference, normalize_label
    from osr_static.oracle import oracle_tau
    from osr_static.partitions import find_record, load_partitions, materialize
    from osr_static.scores import linear_quantile
    from osr_static.data import LabelVault

    document = load_partitions(Path(job["partition_file"]))
    part = find_record(document, int(job["partition_id"]), float(job["nominal_openness"]))
    x, y = load_txt_any(dataset_file(job["data_file"]))
    arrays = materialize(x, y, part)
    dataset = job["dataset"]
    y_tr = np.asarray(arrays["y_tr"]).astype(str)
    grow_idx, cal_idx = holdout_indices(y_tr, float(job["holdout_fraction"]), int(job["seed"]))
    x_grow = arrays["X_tr"][grow_idx]
    y_grow = y_tr[grow_idx]
    x_cal = arrays["X_tr"][cal_idx]
    n_classes = len(np.unique(y_grow))
    dim = int(x_grow.shape[1])
    max_nodes = 10000 * n_classes

    np.random.seed(int(job["seed"]))
    started = time.perf_counter()
    model = loop.GNG_Funciones.create_gng(
        n_i=dim,
        step_gng=0.2,
        max=max_nodes,
        max_age=10,
        n_start=2,
        min_distance=0.001,
    )
    raw_nodes, _model = loop.GNG_Funciones.train_gng(x_grow, model, epochs=int(job["epochs"]))
    train_seconds = time.perf_counter() - started
    nodes = np.asarray(raw_nodes, dtype=float).reshape(-1, dim)

    if dataset == "LETTER":
        y_grow_num = np.asarray(loop.GNG_Funciones.target_to_numeric(y_grow.tolist()))
    else:
        y_grow_num = np.asarray([int(v) for v in y_grow], dtype=int)
    legacy_labels = loop.GNG_Funciones.funcion_comparativaREAL(
        n_classes, x_grow, y_grow_num, nodes, metrica="KNN5_GNG"
    )
    legacy_labels = np.rint(np.asarray(legacy_labels, dtype=float)).astype(int)
    try:
        grouped, _index, ordered, _prop = loop.GNG_Funciones.filtrarGNGREALES(nodes, legacy_labels, n_classes)
        nodes, legacy_labels = loop.GNG_Funciones.from_list_t_arrayREAL(grouped, dim, ordered, n_classes)
        legacy_labels = np.rint(np.asarray(legacy_labels, dtype=float)).astype(int)
    except (IndexError, ValueError):
        legacy_labels = np.rint(np.asarray(legacy_labels, dtype=float)).astype(int)

    scaler_mean = np.asarray(arrays["X_fit"], dtype=float).mean(axis=0)
    scaler_scale = np.asarray(arrays["X_fit"], dtype=float).std(axis=0)
    scaler_scale[scaler_scale == 0] = 1.0

    def scale(matrix):
        return (np.asarray(matrix, dtype=float) - scaler_mean) / scaler_scale

    nodes_s = scale(nodes)
    y_grow_for_nn = to_numeric(dataset, y_grow)
    relabeled = relabel_from_nearest(nodes_s, scale(x_grow), y_grow_for_nn)

    known = [str(v) for v in arrays["kkc_labels_all"]]
    openness = float(100.0 * part["actual_openness"])
    y_true_raw = np.asarray(arrays["y_te"]).astype(str)
    y_true = np.array([normalize_label(v) for v in y_true_raw], dtype=object)

    def pack(name: str, labels_num: np.ndarray) -> dict[str, Any]:
        dist_te, conf_te, pred_te = knn_predict(nodes_s, labels_num, scale(arrays["X_te"]), 5)
        dist_cal, _, _ = knn_predict(nodes_s, labels_num, scale(x_cal), 5)
        pred_s = closed_labels(dataset, pred_te)
        cut = float(linear_quantile(dist_cal, COVERAGE))
        if openness <= 0:
            guessed = pred_s
        else:
            guessed = pred_s.copy()
            guessed[dist_te > cut] = "UUC"
        parts = decompose(y_true, guessed, pred_s, known)
        payload = {
            "paper_f1_percent": float(100.0 * micro_f1_reference(y_true, guessed, known)),
            "FRR": float(parts["FRR"]),
            "FAR": None if parts["n_unknown"] == 0 else float(parts["FAR"]),
            "closed_acc": float(parts["closed_acc"]),
            "dist_threshold": cut,
        }
        if openness > 0:
            score = dist_te / max(cut, 1e-12)
            oracle = oracle_tau(LabelVault(y_true, known), score, pred_s)
            payload["oracle_f1_percent"] = oracle["f1_percent"]
            payload["regret"] = float(oracle["f1_percent"] - payload["paper_f1_percent"])
        return payload

    legacy_rule = pack("legacy_labels", legacy_labels)
    relabel_rule = pack("relabel", relabeled)
    Path(job["out_npz"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(job["out_npz"]),
        node_positions=nodes,
        node_labels_legacy=legacy_labels,
        node_labels_relabel=relabeled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        grow_idx=grow_idx,
        cal_idx=cal_idx,
    )
    return {
        "ok": True,
        "dataset": dataset,
        "partition_id": int(job["partition_id"]),
        "nominal_openness": float(job["nominal_openness"]),
        "actual_openness_percent": openness,
        "epochs": int(job["epochs"]),
        "holdout_fraction": float(job["holdout_fraction"]),
        "seed": int(job["seed"]),
        "config_hash": job["config_hash"],
        "n_grow": int(grow_idx.size),
        "n_cal": int(cal_idx.size),
        "n_nodes": int(nodes.shape[0]),
        "max_nodes": int(max_nodes),
        "train_seconds": float(train_seconds),
        "label_agreement": float(np.mean(relabeled == legacy_labels)),
        "legacy_cal_q95": legacy_rule,
        "relabel_cal_q95": relabel_rule,
    }


def _launch(garage: Path, job: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    out_json = Path(job["out_json"])
    if out_json.is_file():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
        if existing.get("ok"):
            print(f"resume {job['dataset']} p{job['partition_id']} O{job['nominal_openness']}", flush=True)
            return existing
    spec = out_json.with_suffix(".job.json")
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps(job), encoding="utf-8")
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
            [str(garage), "-m", "osr_static.holdout_worker", "--job", str(spec)],
            cwd=str(_repo_root()),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0 or not out_json.is_file():
        return {
            "ok": False,
            "dataset": job["dataset"],
            "partition_id": job["partition_id"],
            "nominal_openness": job["nominal_openness"],
            "returncode": completed.returncode,
            "log": str(log_path),
        }
    return json.loads(out_json.read_text(encoding="utf-8"))


def write_holdout_report(rows: list[dict]) -> None:
    from osr_static.improvements import _fmt, _mean_std

    keys = sorted({(r["dataset"], round(float(r["nominal_openness"]), 2)) for r in rows if r.get("ok")})
    lines = [
        "# Holdout GNG",
        "",
        "One new network per partition. It is grown on 80% of the known-class training rows.",
        "The distance cut is the 0.95 quantile of the other 20%, which never entered the graph.",
        "The confidence gate is off. `legacy_cal_q95` keeps the class-count node vote.",
        "`relabel_cal_q95` labels each node from its nearest grow-set row.",
        "The oracle uses test labels and is not a deployable cut.",
        "",
        "| Dataset | O | n | Legacy labels F1 | Relabel F1 | Legacy FRR | Legacy FAR | Relabel FRR | Relabel FAR | Legacy oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ok = [r for r in rows if r.get("ok")]
    for dataset, nominal in keys:
        group = [r for r in ok if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        cell = {
            "dataset": dataset,
            "nominal_openness": nominal,
            "n": len(group),
            "legacy_f1": _fmt(*_mean_std(r["legacy_cal_q95"]["paper_f1_percent"] for r in group)),
            "relabel_f1": _fmt(*_mean_std(r["relabel_cal_q95"]["paper_f1_percent"] for r in group)),
            "legacy_frr": _fmt(*_mean_std(r["legacy_cal_q95"]["FRR"] for r in group), 3),
            "legacy_far": _fmt(*_mean_std(r["legacy_cal_q95"]["FAR"] for r in group), 3),
            "relabel_frr": _fmt(*_mean_std(r["relabel_cal_q95"]["FRR"] for r in group), 3),
            "relabel_far": _fmt(*_mean_std(r["relabel_cal_q95"]["FAR"] for r in group), 3),
            "oracle": _fmt(*_mean_std(r["legacy_cal_q95"].get("oracle_f1_percent") for r in group)),
        }
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {n} | {legacy_f1} | {relabel_f1} | "
            "{legacy_frr} | {legacy_far} | {relabel_frr} | {relabel_far} | {oracle} |".format(**cell)
        )
    failed = [r for r in rows if not r.get("ok")]
    if failed:
        lines.extend(["", "## Failed", ""])
        for record in failed:
            lines.append(
                f"- {record.get('dataset')} p{record.get('partition_id')} "
                f"O{record.get('nominal_openness')}: {record.get('log', record.get('error', 'failed'))}"
            )
    lines.append("")
    path = RESULTS_STATIC / "HOLDOUT.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def run_holdout(epochs: int = 50, jobs: int = 1, include_closed: bool = False, garage: Path | None = None) -> dict[str, Any]:
    """Launch garage workers. Returns when every selected partition has finished or failed."""
    garage = garage or (_repo_root() / "garage" / "bin" / "python")
    if not garage.is_file():
        raise SystemExit(
            f"garage python not found at {garage}. "
            "The holdout trains a GNG and needs the NeuPy environment used by run.sh. "
            "The baseline virtualenv cannot run this step."
        )
    selected = _jobs(epochs, include_closed, HOLDOUT_FRACTION)
    if not selected:
        raise SystemExit("no holdout jobs selected")
    log_dir = RESULTS_STATIC / "holdout" / "logs" / time.strftime("run_%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(selected)} holdout jobs, {jobs} workers, epochs={epochs}, "
        f"include_closed={include_closed}",
        flush=True,
    )
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        futures = [pool.submit(_launch, garage, job, log_dir) for job in selected]
        for future in as_completed(futures):
            rows.append(future.result())
    write_holdout_report(rows)
    failed = sum(1 for row in rows if not row.get("ok"))
    return {"n": len(rows), "failed": failed, "report": str(RESULTS_STATIC / "HOLDOUT.md")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train holdout GNGs and cut distance on the held-out rows.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--include-closed", action="store_true")
    parser.add_argument("--garage", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    del args.resume
    garage = Path(args.garage) if args.garage else None
    result = run_holdout(args.epochs, args.jobs, args.include_closed, garage)
    print(json.dumps(result), flush=True)
    if int(result.get("failed", 0)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
