"""Train the local baselines on the shared partitions."""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from osr_static.baselines import (
    evm,
    knn_dist,
    knnens,
    libsvm_openset,
    minas,
    one_vs_set,
    osnn,
    pisvm,
    sencforest,
    srosr,
    svm_prob,
    wsvm,
)
from osr_static.baselines.stream_common import map_novelty_predictions
from osr_static.data import LabelVault, load_txt_any
from osr_static.jobs import atomic_json, atomic_npz, config_hash, dataset_file, derive_seed, partition_file
from osr_static.metrics import apply_threshold, micro_f1_reference, operating_report
from osr_static.oracle import oracle_tau
from osr_static.partitions import find_record, load_partitions, materialize
from osr_static.selection import select_threshold

SVM_GRID = (
    (0.1, "scale"), (0.1, 0.01), (0.1, 0.1), (0.1, 1.0),
    (1.0, "scale"), (1.0, 0.01), (1.0, 0.1), (1.0, 1.0),
    (10.0, "scale"), (10.0, 0.01), (10.0, 0.1), (10.0, 1.0),
    (100.0, "scale"), (100.0, 0.01), (100.0, 0.1), (100.0, 1.0),
)
KNN_GRID = (1, 3, 5)
OSNN_THRESHOLDS = np.linspace(0.5, 1.0, 51)
EVM_TAUS = (20, 50, 75)
ONE_VS_SET_GRID = ((1.0, "scale"), (1.0, 0.1), (10.0, "scale"), (10.0, 0.1))
PI_SVM_GRID = ((0.5, "scale"), (1.0, "scale"), (2.0, "scale"), (1.0, 0.1))
WSVM_GRID = ((1.0, "scale"), (10.0, "scale"), (1.0, 0.1))
WOSVM_GRID = ((0.05, "scale"), (0.01, "scale"), (0.05, 0.1))
SROSR_GRID = (0.90, 0.95)
MINAS_GRID = ((3, 1.5), (3, 2.0), (5, 2.0), (5, 2.5), (8, 2.0))
KNNENS_BAG_GRID = ((3, 3), (3, 5), (5, 5))
KNNENS_L = (3, 5)
KNNENS_THRESHOLDS = np.linspace(0.5, 2.5, 21)
SENC_GRID = ((50, 128, 10), (100, 128, 10), (100, 200, 10), (100, 256, 10))


from osr_static.paths import REPO_ROOT, RESULTS_STATIC, benchmark_work_root


def _exp_root() -> Path:
    return REPO_ROOT


def _choose(scored: list[tuple[float, dict, float | None]]) -> tuple[dict, float | None]:
    """Highest validation F1. A tie prefers the threshold closer to 1 when ``tau`` is used."""

    def _key(item: tuple[float, dict, float | None]) -> tuple[float, float]:
        tau = item[2]
        tie = -abs(tau - 1.0) if tau is not None else 0.0
        return (item[0], tie)

    best_f1, best_params, best_tau = max(scored, key=_key)
    return best_params, best_tau


def _osnn_select(x_fit, y_fit, x_val, y_val, known, rejection_off: bool):
    scores, pred = osnn.scores(x_fit, y_fit, x_val)
    if rejection_off:
        return {}, None
    best = (-1.0, 0.5)
    for threshold in OSNN_THRESHOLDS:
        guessed = apply_threshold(scores, pred, float(threshold))
        f1 = micro_f1_reference(y_val, guessed, known)
        closer = abs(float(threshold) - 0.75) < abs(best[1] - 0.75)
        if f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and closer):
            best = (float(f1), float(threshold))
    return {"T": best[1]}, best[1]


def _knn_select(x_fit, y_fit, x_val, y_val, known, rejection_off: bool):
    candidates = []
    for k in KNN_GRID:
        scores, pred = knn_dist.scores(x_fit, y_fit, x_val, k)
        if rejection_off:
            f1 = micro_f1_reference(y_val, pred, known)
            candidates.append((f1, {"k": int(k)}, 1.0))
        else:
            tau, f1 = select_threshold(scores, pred, y_val, known)
            candidates.append((f1, {"k": int(k)}, tau))
    return _choose(candidates)


def _svm_select(x_fit, y_fit, x_val, y_val, known, rejection_off: bool, grid):
    candidates = []
    for c_value, gamma in grid:
        scores, pred = svm_prob.fit_predict(x_fit, y_fit, x_val, c_value, gamma)
        params = {"C": c_value, "gamma": gamma}
        if rejection_off:
            f1 = micro_f1_reference(y_val, pred, known)
            candidates.append((f1, params, 1.0))
        else:
            tau, f1 = select_threshold(scores, pred, y_val, known)
            candidates.append((f1, params, tau))
    return _choose(candidates)


def _grid_select(fit_predict, grid, x_fit, y_fit, x_val, y_val, known, rejection_off: bool, key_names: tuple[str, ...]):
    candidates = []
    for values in grid:
        params = {name: value for name, value in zip(key_names, values if isinstance(values, tuple) else (values,))}
        scores, pred = fit_predict(x_fit, y_fit, x_val, **params)
        if rejection_off:
            f1 = micro_f1_reference(y_val, pred, known)
            candidates.append((f1, params, 1.0))
        else:
            tau, f1 = select_threshold(scores, pred, y_val, known)
            candidates.append((f1, params, tau))
    return _choose(candidates)


def _minas_select(x_fit, y_fit, x_val, y_val, known, rejection_off: bool):
    candidates = []
    for k_micro, radius_factor in MINAS_GRID:
        model = minas.fit(x_fit, y_fit, k_micro, radius_factor)
        scores, pred = minas.scores(model, x_val)
        params = {"k_micro": int(k_micro), "radius_factor": float(radius_factor)}
        if rejection_off:
            f1 = micro_f1_reference(y_val, pred, known)
            candidates.append((f1, params, 1.0))
        else:
            y_pred = apply_threshold(scores, pred, 1.0)
            f1 = micro_f1_reference(y_val, y_pred, known)
            candidates.append((f1, params, 1.0))
    return _choose(candidates)


def _knnens_select(x_fit, y_fit, x_val, y_val, known, rejection_off: bool):
    candidates = []
    for b, a in KNNENS_BAG_GRID:
        for L in KNNENS_L:
            model = knnens.fit(x_fit, y_fit, b, a, L)
            if rejection_off:
                scores, pred = knnens.scores(model, x_val, 1e9)
                f1 = micro_f1_reference(y_val, pred, known)
                candidates.append((f1, {"b": b, "a": a, "L": L, "t": 1e9}, 1.0))
                continue
            best = (-1.0, 0.9)
            for threshold in KNNENS_THRESHOLDS:
                _scores, pred = knnens.scores(model, x_val, float(threshold))
                f1 = micro_f1_reference(y_val, map_novelty_predictions(pred), known)
                if f1 > best[0] + 1e-12:
                    best = (float(f1), float(threshold))
            candidates.append((best[0], {"b": b, "a": a, "L": L, "t": best[1]}, None))
    return _choose(candidates)


def _senc_select(x_fit, y_fit, x_val, y_val, known, rejection_off: bool):
    candidates = []
    for n_estimators, max_samples, min_size in SENC_GRID:
        model = sencforest.fit(x_fit, y_fit, n_estimators, max_samples, min_size)
        scores, pred = sencforest.scores(model, x_val)
        params = {
            "n_estimators": int(n_estimators),
            "max_samples": int(max_samples),
            "min_size": int(min_size),
        }
        if rejection_off:
            f1 = micro_f1_reference(y_val, pred, known)
            candidates.append((f1, params, 1.0))
        else:
            f1 = micro_f1_reference(y_val, map_novelty_predictions(pred), known)
            candidates.append((f1, params, None))
    return _choose(candidates)


def _evm_select(x_fit, y_fit, x_val, y_val, known, rejection_off: bool):
    candidates = []
    for tau_evm in EVM_TAUS:
        model = evm.fit(x_fit, y_fit, tau_evm)
        scores, pred = evm.scores(model, x_val)
        params = {"tau_evm": int(tau_evm), "fit_failures": int(model["failures"])}
        if rejection_off:
            f1 = micro_f1_reference(y_val, pred, known)
            candidates.append((f1, params, 1.0))
        else:
            tau, f1 = select_threshold(scores, pred, y_val, known)
            candidates.append((f1, params, tau))
    return _choose(candidates)


def execute_baseline(job: dict[str, Any]) -> dict[str, Any]:
    """Train one baseline on one partition and write its JSON."""
    out_json = Path(job["out_json"])
    if out_json.exists():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
        if existing.get("ok"):
            return existing
    started = time.time()
    try:
        record = _execute(job)
    except Exception as exc:
        failed = {"ok": False, "error": str(exc), "traceback": traceback.format_exc(), "job": job}
        atomic_json(job["failed_json"], failed)
        return failed
    record["wall_seconds"] = time.time() - started
    atomic_json(out_json, record)
    return record


def _execute(job: dict[str, Any]) -> dict[str, Any]:
    document = load_partitions(Path(job["partition_file"]))
    part = find_record(document, int(job["partition_id"]), float(job["nominal_openness"]))
    x, y = load_txt_any(dataset_file(job["data_file"]))
    arrays = materialize(x, y, part)
    scaler = StandardScaler().fit(arrays["X_fit"])
    x_fit = scaler.transform(arrays["X_fit"])
    x_val = scaler.transform(arrays["X_val_open"])
    x_tr = scaler.transform(arrays["X_tr"])
    x_te = scaler.transform(arrays["X_te"])
    y_fit = np.asarray(arrays["y_fit"]).astype(str)
    y_val = np.asarray(arrays["y_val_open"]).astype(str)
    y_tr = np.asarray(arrays["y_tr"]).astype(str)
    y_te = np.asarray(arrays["y_te"]).astype(str)
    known_fit = [str(v) for v in arrays["kkc_labels_fit"]]
    known = [str(v) for v in arrays["kkc_labels_all"]]
    rejection_off = float(part["actual_openness"]) <= 0
    method = job["method"]
    grid = SVM_GRID[:4] if job.get("svm_reduced") else SVM_GRID
    t0 = time.perf_counter()
    if method == "osnn":
        params, tau = _osnn_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off)
        scores, pred = osnn.scores(x_tr, y_tr, x_te)
    elif method == "knn_dist":
        params, tau = _knn_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off)
        scores, pred = knn_dist.scores(x_tr, y_tr, x_te, int(params["k"]))
        if rejection_off:
            tau = None
    elif method == "svm_prob":
        params, tau = _svm_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off, grid)
        scores, pred = svm_prob.fit_predict(x_tr, y_tr, x_te, params["C"], params["gamma"])
        if rejection_off:
            tau = None
    elif method == "evm":
        params, tau = _evm_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off)
        model = evm.fit(x_tr, y_tr, int(params["tau_evm"]))
        scores, pred = evm.scores(model, x_te)
        params = dict(params)
        params["test_fit_failures"] = int(model["failures"])
        if rejection_off:
            tau = None
    elif method == "one_vs_set":
        grid = ONE_VS_SET_GRID[:1] if job.get("svm_reduced") else ONE_VS_SET_GRID
        params, tau = _grid_select(one_vs_set.fit_predict, grid, x_fit, y_fit, x_val, y_val, known_fit, rejection_off, ("C", "gamma"))
        scores, pred = one_vs_set.fit_predict(x_tr, y_tr, x_te, params["C"], params["gamma"])
        if rejection_off:
            tau = None
    elif method == "pisvm":
        grid = PI_SVM_GRID[:1] if job.get("svm_reduced") else PI_SVM_GRID
        params, tau = _grid_select(pisvm.fit_predict, grid, x_fit, y_fit, x_val, y_val, known_fit, rejection_off, ("C", "gamma"))
        scores, pred = pisvm.fit_predict(x_tr, y_tr, x_te, params["C"], params["gamma"])
        if rejection_off:
            tau = None
    elif method == "wsvm":
        grid = WSVM_GRID[:1] if job.get("svm_reduced") else WSVM_GRID
        params, tau = _grid_select(wsvm.fit_predict_wsvm, grid, x_fit, y_fit, x_val, y_val, known_fit, rejection_off, ("C", "gamma"))
        scores, pred = wsvm.fit_predict_wsvm(x_tr, y_tr, x_te, params["C"], params["gamma"])
        if rejection_off:
            tau = None
    elif method == "wosvm":
        grid = WOSVM_GRID[:1] if job.get("svm_reduced") else WOSVM_GRID
        params, tau = _grid_select(wsvm.fit_predict_wosvm, grid, x_fit, y_fit, x_val, y_val, known_fit, rejection_off, ("nu", "gamma"))
        scores, pred = wsvm.fit_predict_wosvm(x_tr, y_tr, x_te, params["nu"], params["gamma"])
        if rejection_off:
            tau = None
    elif method == "srosr":
        grid = SROSR_GRID[:1] if job.get("svm_reduced") else SROSR_GRID
        params, tau = _grid_select(srosr.fit_predict, grid, x_fit, y_fit, x_val, y_val, known_fit, rejection_off, ("variance_keep",))
        scores, pred = srosr.fit_predict(x_tr, y_tr, x_te, params["variance_keep"])
        if rejection_off:
            tau = None
    elif method == "minas":
        params, tau = _minas_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off)
        model = minas.fit(x_tr, y_tr, params["k_micro"], params["radius_factor"])
        scores, pred = minas.scores(model, x_te)
        if rejection_off:
            tau = None
        else:
            tau = 1.0
    elif method == "knnens":
        params, tau = _knnens_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off)
        model = knnens.fit(x_tr, y_tr, params["b"], params["a"], params["L"])
        t_apply = 1e9 if rejection_off else params["t"]
        scores, pred = knnens.scores(model, x_te, t_apply)
        pred = map_novelty_predictions(pred)
        tau = None
    elif method == "sencforest":
        params, tau = _senc_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off)
        model = sencforest.fit(
            x_tr,
            y_tr,
            params["n_estimators"],
            params["max_samples"],
            params["min_size"],
        )
        scores, pred = sencforest.scores(model, x_te)
        pred = map_novelty_predictions(pred)
        tau = None
    else:
        raise ValueError(method)
    train_seconds = time.perf_counter() - t0
    vault = LabelVault(y_te, known)
    report = operating_report(vault, scores, pred, tau)
    oracle = None if rejection_off or tau is None else oracle_tau(vault, scores, pred)
    atomic_npz(
        Path(job["out_npz"]),
        scores=np.asarray(scores, dtype=float),
        pred=np.asarray(pred).astype(str),
        oscr_fpr=report["oscr_fpr"],
        oscr_ccr=report["oscr_ccr"],
    )
    return {
        "ok": True,
        "experiment": "baselines",
        "dataset": job["dataset"],
        "partition_id": int(job["partition_id"]),
        "nominal_openness": float(job["nominal_openness"]),
        "actual_openness_percent": float(100.0 * part["actual_openness"]),
        "method": method,
        "seed": int(job["seed"]),
        "config_hash": job["config_hash"],
        "selected_params": params,
        "threshold": tau,
        "rejection_off": rejection_off,
        "paper_f1_percent": report["paper_f1_percent"],
        "repo_f1_percent": report["repo_f1_percent"],
        "auroc": report["auroc"],
        "aupr": report["aupr"],
        "oscr": report["oscr"],
        "decomposition": report["decomposition"],
        "oracle_tau_f1": None if oracle is None else oracle["f1_percent"],
        "oracle_tau": None if oracle is None else oracle["tau"],
        "regret": None if oracle is None else float(oracle["f1_percent"] - report["paper_f1_percent"]),
        "train_seconds": train_seconds,
        "libsvm_openset": libsvm_openset.available()[1],
    }


def enumerate_jobs(config: dict, tag: str = "baselines") -> list[dict]:
    exp = _exp_root()
    methods = list(config.get("methods", ["osnn", "knn_dist", "svm_prob"]))
    hashed = config_hash({"methods": methods, "svm_reduced": bool(config.get("svm_reduced", False)), "split_seed": 123})
    jobs = []
    for dataset, spec in config["datasets"].items():
        part_path = partition_file(dataset)
        document = load_partitions(part_path)
        data_path = dataset_file(spec["file"])
        for part in document["partitions"]:
            for method in methods:
                nominal = float(part["nominal_openness"])
                partition_id = int(part["partition_id"])
                nominal_name = f"{nominal:.2f}"
                out_dir = benchmark_work_root() / tag / "raw" / dataset / f"p{partition_id}" / f"O{nominal_name}"
                stem = out_dir / method
                jobs.append({
                    "dataset": dataset,
                    "data_file": spec["file"],
                    "partition_file": str(part_path),
                    "partition_id": partition_id,
                    "nominal_openness": nominal,
                    "method": method,
                    "seed": derive_seed(dataset, partition_id, nominal, method, hashed),
                    "config_hash": hashed,
                    "repo_root": str(REPO_ROOT),
                    "svm_reduced": bool(config.get("svm_reduced", False)),
                    "out_json": str(stem.with_suffix(".json")),
                    "out_npz": str(stem.with_suffix(".npz")),
                    "failed_json": str(
                        benchmark_work_root() / tag / "failed" / f"{dataset}_p{partition_id}_O{nominal_name}_{method}.json"
                    ),
                })
    return jobs


def run_baselines(config: dict, only: dict[str, str] | None, n_jobs: int, tag: str = "baselines") -> dict:
    jobs = enumerate_jobs(config, tag=tag)
    if only:
        jobs = [
            job for job in jobs
            if ("dataset" not in only or job["dataset"].lower() == only["dataset"].lower())
            and ("openness" not in only or abs(job["nominal_openness"] - float(only["openness"])) < 1e-6)
            and ("partition" not in only or job["partition_id"] == int(only["partition"]))
            and ("method" not in only or job["method"].lower() == only["method"].lower())
        ]
    if not jobs:
        raise SystemExit("no baseline jobs selected")
    print(f"{len(jobs)} baseline jobs, {n_jobs} workers", flush=True)
    records = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        futures = [pool.submit(execute_baseline, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if record.get("ok"):
                print(
                    f"done {record['method']} {record['dataset']} p{record['partition_id']} "
                    f"O{record['nominal_openness']} F1={record['paper_f1_percent']:.2f} "
                    f"in {record['train_seconds']:.0f}s",
                    flush=True,
                )
            else:
                print("FAILED " + str(record.get("error")), flush=True)
    summary = {
        "ok_jobs": sum(1 for record in records if record.get("ok")),
        "failed_jobs": sum(1 for record in records if not record.get("ok")),
        "wall_seconds": time.time() - started,
    }
    atomic_json(benchmark_work_root() / tag / "summary.json", summary)
    if summary["failed_jobs"]:
        raise SystemExit(f"{summary['failed_jobs']} baseline jobs failed")
    return summary
