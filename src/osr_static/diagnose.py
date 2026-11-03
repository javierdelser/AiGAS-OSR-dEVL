"""E1, E2 and E3 for the saved GNG + 1_NN models.

No GNG is trained here. Radius scores are Euclidean distances to class centroids
of the scaled training rows. The GNG triple in the legacy script is a single
candidate, so the sequential and joint q selections are the same number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from osr_static.data import LabelVault, load_txt_any
from osr_static.jobs import atomic_json, atomic_npz
from osr_static.metrics import apply_threshold, micro_f1_reference, operating_report
from osr_static.oracle import oracle_q, oracle_tau
from osr_static.partitions import find_record, load_partitions, materialize
from osr_static.scores import class_centroids, radius_scores
from osr_static.selection import Q_GRID, f1_at_tau, select_q

REJECT_TAU = 1.0


from osr_static.paths import DATA_STATIC, REPO_ROOT, RESULTS_STATIC


def _exp_root() -> Path:
    return REPO_ROOT


def _repo_root() -> Path:
    return REPO_ROOT


def _scale(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=float).copy()
    scale[scale == 0] = 1.0
    return (np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)) / scale


def _closed_labels(dataset: str, pred_base: np.ndarray) -> np.ndarray:
    labels = []
    for value in np.asarray(pred_base).ravel():
        number = int(value)
        if number == 100:
            labels.append("UUC")
        elif dataset == "LETTER":
            labels.append(chr(number - 1 + ord("A")))
        else:
            labels.append(str(number))
    return np.asarray(labels, dtype=object)


def _phase1_records(phase1: Path) -> list[tuple[dict, Path]]:
    found = []
    for path in sorted(phase1.glob("raw/phase1/*/*/*/1_NN__*.json")):
        if path.name.endswith(".job.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("ok"):
            found.append((record, path))
    return found


def _arrays_for_q(x_ref, y_ref, x_query, q: float):
    centroids, radii, labels, small = class_centroids(x_ref, y_ref, q)
    s_all, s_nn, s_raw, pred = radius_scores(x_query, centroids, radii, labels)
    return s_all, s_nn, s_raw, pred, small


def analyze_record(record: dict, json_path: Path, bundle: dict[str, Any], q_grid: tuple[float, ...]) -> tuple[dict, dict]:
    """One partition: threshold-free 1_NN metrics, radius q selection, oracle sweep."""
    dataset = record["dataset"]
    arrays = materialize(
        bundle["x"],
        bundle["y"],
        find_record(bundle["document"], record["partition_id"], record["nominal_openness"]),
    )
    with np.load(json_path.with_suffix(".npz"), allow_pickle=True) as saved:
        score = np.asarray(saved["score"], dtype=float)
        y_true = np.asarray(saved["y_true"]).astype(str)
        y_pred = np.asarray(saved["y_pred"]).astype(str)
        pred_base = _closed_labels(dataset, saved["pred_base"])
        mean = np.asarray(saved["scaler_mean"], dtype=float)
        scale = np.asarray(saved["scaler_scale"], dtype=float)

    known = [str(v) for v in record["known_classes"]]
    fitting = [str(v) for v in record["fitting_classes"]]
    paper = micro_f1_reference(y_true, y_pred, known)
    if abs(100.0 * paper - float(record["paper_f1_percent"])) > 1e-6:
        raise RuntimeError(f"stored paper F1 does not match the saved predictions for {json_path}")

    openness = float(record["actual_openness_percent"])
    x_fit = _scale(arrays["X_fit"], mean, scale)
    x_val = _scale(arrays["X_val_open"], mean, scale)
    x_tr = _scale(arrays["X_tr"], mean, scale)
    x_te = _scale(arrays["X_te"], mean, scale)
    y_fit = np.asarray(arrays["y_fit"]).astype(str)
    y_tr = np.asarray(arrays["y_tr"]).astype(str)
    y_val = np.asarray(arrays["y_val_open"]).astype(str)

    vault = LabelVault(y_true, known)
    reject_tau = None if openness <= 0 else REJECT_TAU
    primary = operating_report(vault, score, pred_base, reject_tau)
    if openness > 0:
        mismatch = int(np.sum(apply_threshold(score, pred_base, REJECT_TAU) != y_pred))
    else:
        mismatch = int(np.sum(pred_base != y_pred))

    val_f1: dict[float, float] = {}
    test_f1: dict[float, float] = {}
    test_scores: dict[float, dict] = {}
    small_train = 0
    for q in q_grid:
        _s_all, s_val, _raw, pred_val, _small = _arrays_for_q(x_fit, y_fit, x_val, q)
        val_f1[q] = f1_at_tau(s_val, pred_val, y_val, fitting, REJECT_TAU)
        s_all, s_nn, s_raw, pred_te, small_train = _arrays_for_q(x_tr, y_tr, x_te, q)
        test_scores[q] = {"s_all": s_all, "s_nn": s_nn, "s_raw": s_raw, "pred": pred_te}
        guessed = pred_te if openness <= 0 else apply_threshold(s_nn, pred_te, REJECT_TAU)
        test_f1[q] = micro_f1_reference(y_true, guessed, known)

    chosen = select_q(val_f1)
    q85 = min(q_grid, key=lambda q: abs(q - 0.85))
    radius_85 = test_scores[q85]
    radius_sel = test_scores[chosen]
    report_85 = operating_report(vault, radius_85["s_nn"], radius_85["pred"], reject_tau)
    report_all = operating_report(vault, radius_85["s_all"], radius_85["pred"], None) if openness > 0 else None
    report_raw = operating_report(vault, radius_85["s_raw"], radius_85["pred"], None) if openness > 0 else None
    oracle_primary = oracle_tau(vault, score, pred_base) if openness > 0 else None
    oracle_85 = oracle_tau(vault, radius_85["s_nn"], radius_85["pred"]) if openness > 0 else None
    oracle_sel = oracle_tau(vault, radius_sel["s_nn"], radius_sel["pred"]) if openness > 0 else None
    q_oracle = oracle_q(vault, {q: test_scores[q]["s_nn"] for q in q_grid}, radius_sel["pred"]) if openness > 0 else None

    auroc_by_q = {}
    if openness > 0:
        for q in q_grid:
            auroc_by_q[f"{q:.3f}"] = operating_report(vault, test_scores[q]["s_nn"], test_scores[q]["pred"], None)["auroc"]

    actual_1nn = float(record["paper_f1_percent"])
    payload = {
        "ok": True,
        "dataset": dataset,
        "partition_id": int(record["partition_id"]),
        "nominal_openness": float(record["nominal_openness"]),
        "actual_openness_percent": openness,
        "method": "1_NN",
        "radius_variant": "s_nn",
        "joint_equals_sequential": True,
        "rejection_off": openness <= 0,
        "tau_match_mismatches": mismatch,
        "n_small_train_classes": int(small_train),
        "auroc_1nn": primary["auroc"],
        "aupr_1nn": primary["aupr"],
        "oscr_1nn": primary["oscr"],
        "auroc_s_all_q85": None if report_all is None else report_all["auroc"],
        "aupr_s_all_q85": None if report_all is None else report_all["aupr"],
        "oscr_s_all_q85": None if report_all is None else report_all["oscr"],
        "auroc_s_nn_q85": report_85["auroc"],
        "aupr_s_nn_q85": report_85["aupr"],
        "oscr_s_nn_q85": report_85["oscr"],
        "auroc_s_raw": None if report_raw is None else report_raw["auroc"],
        "aupr_s_raw": None if report_raw is None else report_raw["aupr"],
        "oscr_s_raw": None if report_raw is None else report_raw["oscr"],
        "auroc_by_q": auroc_by_q,
        "decomposition_1nn": record["decomposition"],
        "decomposition_radius_q85": report_85["decomposition"],
        "val_f1_by_q": {f"{q:.3f}": float(100.0 * val_f1[q]) for q in q_grid},
        "test_f1_radius_by_q": {f"{q:.3f}": float(100.0 * test_f1[q]) for q in q_grid},
        "selected_q": None if openness <= 0 else chosen,
        "e2_sequential_f1": actual_1nn if openness <= 0 else float(100.0 * test_f1[chosen]),
        "e2_joint_f1": actual_1nn if openness <= 0 else float(100.0 * test_f1[chosen]),
        "radius_q85_f1": float(100.0 * test_f1[q85]),
        "actual_f1_1nn": actual_1nn,
        "oracle_tau_1nn": None if oracle_primary is None else oracle_primary["f1_percent"],
        "oracle_tau_1nn_threshold": None if oracle_primary is None else oracle_primary["tau"],
        "oracle_tau_radius_q85": None if oracle_85 is None else oracle_85["f1_percent"],
        "oracle_tau_radius_selected": None if oracle_sel is None else oracle_sel["f1_percent"],
        "oracle_q_f1": None if q_oracle is None else q_oracle["f1_percent"],
        "oracle_q": None if q_oracle is None else q_oracle["q"],
        "regret_1nn": None if oracle_primary is None else float(oracle_primary["f1_percent"] - actual_1nn),
        "regret_radius_q85": None if oracle_85 is None else float(oracle_85["f1_percent"] - 100.0 * test_f1[q85]),
    }
    curves = {
        "oscr_1nn_fpr": primary["oscr_fpr"],
        "oscr_1nn_ccr": primary["oscr_ccr"],
        "oscr_radius85_fpr": report_85["oscr_fpr"],
        "oscr_radius85_ccr": report_85["oscr_ccr"],
    }
    if oracle_primary is not None:
        curves["f1_curve_1nn"] = oracle_primary["f1_curve"]
        curves["tau_curve_1nn"] = np.asarray(oracle_primary["tau_curve"], dtype=float)
        curves["f1_curve_radius85"] = oracle_85["f1_curve"]
        curves["tau_curve_radius85"] = np.asarray(oracle_85["tau_curve"], dtype=float)
    return payload, curves


def run_diagnostics(config: dict, only: dict[str, str] | None = None, jobs: int = 1) -> dict:
    """Write one JSON per saved Phase 1 model under ``results/diagnostics``."""
    del jobs
    phase1 = RESULTS_STATIC / "phase1"
    records = _phase1_records(phase1)
    if not records:
        raise SystemExit(f"no Phase 1 results in {phase1}")
    q_grid = tuple(float(q) for q in config.get("q_grid", Q_GRID))
    out_root = RESULTS_STATIC / "diagnostics" / "raw"
    cache: dict[str, dict] = {}
    written = 0
    skipped = 0
    for record, path in records:
        if only:
            if "dataset" in only and record["dataset"].lower() != only["dataset"].lower():
                continue
            if "openness" in only and abs(float(record["nominal_openness"]) - float(only["openness"])) > 1e-6:
                continue
            if "partition" in only and int(record["partition_id"]) != int(only["partition"]):
                continue
        nominal = f"{float(record['nominal_openness']):.2f}"
        dest = out_root / record["dataset"] / f"p{record['partition_id']}" / f"O{nominal}" / "1_NN.json"
        if dest.exists():
            skipped += 1
            continue
        dataset = record["dataset"]
        if dataset not in cache:
            spec = config["datasets"][dataset]
            x, y = load_txt_any(DATA_STATIC / spec["file"])
            cache[dataset] = {
                "x": x,
                "y": y,
                "document": load_partitions(exp / "data" / "partitions" / f"{dataset}.json"),
            }
        payload, curves = analyze_record(record, path, cache[dataset], q_grid)
        atomic_json(dest, payload)
        atomic_npz(dest.with_suffix(".npz"), **curves)
        written += 1
        print(
            f"diagnostics {dataset} p{record['partition_id']} O{nominal} "
            f"AUROC={payload['auroc_1nn']} regret={payload['regret_1nn']}",
            flush=True,
        )
    summary = {"written": written, "skipped": skipped, "q_grid": list(q_grid)}
    atomic_json(RESULTS_STATIC / "diagnostics" / "summary.json", summary)
    return summary
