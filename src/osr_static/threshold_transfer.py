"""Check whether the 1_NN distance cut is too loose on the deployed GNG.

Validation picks a distance cut on a network grown from the fitting classes only.
The test decision uses a second network grown from every known-class training row,
and the numeric cut is copied across. ``COBERTURA_CONOCIDAS`` in the legacy code
is 0.95: the cut must leave at least that fraction of validation knowns accepted.

This module does not train a GNG. For each saved Phase 1 network it

1. rebuilds the scaled 5-NN geometry on the deployed nodes,
2. records where the copied distance cut sits in the known-distance distribution
   of that same network,
3. replaces the copied distance cut with the 0.95 quantile of those distances.

The training-row quantile and the fitting-class validation quantile do not use
test labels. Both sets of rows were used to grow the deployed network, so those
quantiles are in-sample. The test-known quantile uses the known test rows to
place the same 0.95 cut and is marked oracle: unknown labels are not used to
choose it, and it is not a deployable threshold. At openness 0 the legacy
script does not select a cut and rejection stays off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from osr_static.data import load_txt_any
from osr_static.jobs import atomic_json, dataset_file, experiment_root, partition_file
from osr_static.paths import RESULTS_STATIC
from osr_static.metrics import decompose, micro_f1_reference, normalize_label
from osr_static.partitions import find_record, load_partitions, materialize
from osr_static.scores import linear_quantile

COVERAGE = 0.95
N_NEIGHBORS = 5
DATA_FILES = {
    "LETTER": "letter_dataset.csv",
    "PENDIGITS": "pendigits_dataset.csv",
}
DIST_ATOL = 1e-4


def empirical_cdf(threshold: float, values: np.ndarray) -> float:
    """Fraction of ``values`` that fall at or below ``threshold``."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values <= threshold))


def looser_than_coverage(threshold: float, reference: np.ndarray, coverage: float = COVERAGE) -> bool:
    """True when ``threshold`` accepts more of ``reference`` than ``coverage``."""
    reference = np.asarray(reference, dtype=float)
    if reference.size == 0 or not np.isfinite(threshold):
        return False
    return bool(threshold > linear_quantile(reference, coverage))


def apply_gates(
    dist_mean: np.ndarray,
    confidence: np.ndarray,
    pred_closed: np.ndarray,
    dist_threshold: float,
    conf_threshold: float,
    openness: float,
) -> np.ndarray:
    """Legacy reject rule. A confidence cut of 0 never fires. Openness 0 never rejects."""
    pred = np.asarray(pred_closed, dtype=object).copy()
    if openness <= 0:
        return pred
    reject = (np.asarray(dist_mean, dtype=float) > dist_threshold) | (
        np.asarray(confidence, dtype=float) < conf_threshold
    )
    pred[reject] = "UUC"
    return pred


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


class _DeployedNeighbors:
    """5-NN on the scaled deployed nodes. One fit, several query sets."""

    def __init__(self, nodes: np.ndarray, node_labels: np.ndarray, mean: np.ndarray, scale: np.ndarray):
        labels = np.rint(np.asarray(node_labels, dtype=float)).astype(int)
        self._nodes = _scale(nodes, mean, scale)
        self._mean = np.asarray(mean, dtype=float)
        self._scale = np.asarray(scale, dtype=float).copy()
        self._scale[self._scale == 0] = 1.0
        k = min(N_NEIGHBORS, self._nodes.shape[0])
        self._knn = KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree")
        self._knn.fit(self._nodes, labels)

    def query(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaled = _scale(raw, self._mean, self._scale)
        distances, _ = self._knn.kneighbors(scaled)
        dist_mean = distances.mean(axis=1)
        pred = np.rint(self._knn.predict(scaled).astype(float)).astype(int)
        confidence = self._knn.predict_proba(scaled).max(axis=1)
        return dist_mean, confidence, pred


def _phase1_records(phase1: Path) -> list[tuple[dict, Path]]:
    found = []
    for path in sorted(phase1.glob("raw/phase1/*/*/*/1_NN__*.json")):
        if path.name.endswith(".job.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("ok"):
            found.append((record, path))
    return found


def _median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _rule_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    pred_closed: np.ndarray,
    known: Sequence[str],
) -> dict[str, float | None]:
    parts = decompose(y_true, pred, pred_closed, known)
    return {
        "paper_f1_percent": float(100.0 * micro_f1_reference(y_true, pred, known)),
        "FRR": float(parts["FRR"]),
        "FAR": None if parts["n_unknown"] == 0 else float(parts["FAR"]),
        "closed_acc": float(parts["closed_acc"]),
    }


def _group_summary(dist: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "n": int(dist.size),
        "median_dist": _median(dist),
        "q95_dist": float(linear_quantile(dist, COVERAGE)) if dist.size else float("nan"),
        "cdf_at_copied_cut": empirical_cdf(threshold, dist),
    }


def analyze_record(record: dict, json_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    """One partition. Raises if the rebuilt 5-NN distances disagree with the saved test scores."""
    dataset = record["dataset"]
    arrays = materialize(
        bundle["x"],
        bundle["y"],
        find_record(bundle["document"], record["partition_id"], record["nominal_openness"]),
    )
    with np.load(json_path.with_suffix(".npz"), allow_pickle=True) as saved:
        saved_dist = np.asarray(saved["dist_mean"], dtype=float)
        saved_conf = np.asarray(saved["confidence"], dtype=float)
        saved_pred = np.asarray(saved["y_pred"]).astype(str)
        y_true = np.asarray(saved["y_true"]).astype(str)
        pred_closed = _closed_labels(dataset, saved["pred_base"])
        nodes = np.asarray(saved["node_positions"], dtype=float)
        node_labels = np.asarray(saved["node_labels"])
        mean = np.asarray(saved["scaler_mean"], dtype=float)
        scale = np.asarray(saved["scaler_scale"], dtype=float)

    openness = float(record["actual_openness_percent"])
    dist_thr = float(record["threshold_dist"])
    conf_thr = float(record["threshold_conf"])
    known = [str(v) for v in record["known_classes"]]
    selected = openness > 0

    geometry = _DeployedNeighbors(nodes, node_labels, mean, scale)
    dist_te, conf_te, pred_te = geometry.query(arrays["X_te"])
    dist_gap = float(np.max(np.abs(dist_te - saved_dist))) if saved_dist.size else 0.0
    conf_gap = float(np.max(np.abs(conf_te - saved_conf))) if saved_conf.size else 0.0
    if dist_gap > DIST_ATOL or conf_gap > DIST_ATOL:
        raise RuntimeError(
            f"rebuilt 5-NN does not match the saved test scores for {json_path} "
            f"(dist {dist_gap}, confidence {conf_gap})"
        )
    label_gap = int(np.sum(_closed_labels(dataset, pred_te) != pred_closed))

    dist_tr, _, _ = geometry.query(arrays["X_tr"])
    dist_val, conf_val, _ = geometry.query(arrays["X_val_open"])
    y_val = np.array([normalize_label(v) for v in arrays["y_val_open"]], dtype=object)
    val_known = y_val != "UUC"
    y_true_n = np.array([normalize_label(v) for v in y_true], dtype=object)
    known_n = np.array([normalize_label(v) for v in known], dtype=object)
    test_known = np.isin(y_true_n, known_n)
    test_unknown = ~test_known

    copied_pred = apply_gates(saved_dist, saved_conf, pred_closed, dist_thr, conf_thr, openness)
    if int(np.sum(copied_pred != np.array([normalize_label(v) for v in saved_pred], dtype=object))) != 0:
        raise RuntimeError(f"copied thresholds do not reproduce the saved predictions for {json_path}")
    copied_f1 = _rule_metrics(y_true, copied_pred, pred_closed, known)
    if abs(copied_f1["paper_f1_percent"] - float(record["paper_f1_percent"])) > 1e-6:
        raise RuntimeError(f"copied paper F1 does not match the stored value for {json_path}")

    reference = {
        "train": dist_tr,
        "val_known": dist_val[val_known],
        "val_simulated": dist_val[~val_known],
        "test_known": saved_dist[test_known],
        "test_unknown": saved_dist[test_unknown],
    }
    groups = {name: _group_summary(dist, dist_thr) for name, dist in reference.items()}

    train_q95 = groups["train"]["q95_dist"]
    val_q95 = groups["val_known"]["q95_dist"]
    test_known_q95 = groups["test_known"]["q95_dist"]

    rules = {
        "copied": (dist_thr, conf_thr),
        "train_q95": (train_q95, conf_thr),
        "train_q95_distance_only": (train_q95, 0.0),
        "val_known_q95": (val_q95, conf_thr),
        "oracle_known_q95": (test_known_q95, conf_thr),
    }
    rule_metrics = {}
    for name, (rule_dist, rule_conf) in rules.items():
        pred = apply_gates(saved_dist, saved_conf, pred_closed, rule_dist, rule_conf, openness)
        metrics = _rule_metrics(y_true, pred, pred_closed, known)
        metrics["dist_threshold"] = None if not np.isfinite(rule_dist) else float(rule_dist)
        metrics["conf_threshold"] = float(rule_conf)
        metrics["uses_test_labels"] = name.startswith("oracle_")
        rule_metrics[name] = metrics

    dist_reject = saved_dist > dist_thr
    conf_reject = saved_conf < conf_thr
    def _rate(mask: np.ndarray, group: np.ndarray) -> float | None:
        if group.sum() == 0:
            return None
        return float(np.mean(mask[group]))

    return {
        "ok": True,
        "dataset": dataset,
        "partition_id": int(record["partition_id"]),
        "nominal_openness": float(record["nominal_openness"]),
        "actual_openness_percent": openness,
        "threshold_was_selected": selected,
        "n_nodes": int(nodes.shape[0]),
        "copied_dist": dist_thr,
        "copied_conf": conf_thr,
        "rebuilt_dist_max_abs": dist_gap,
        "rebuilt_conf_max_abs": conf_gap,
        "rebuilt_label_mismatches": label_gap,
        "copied_looser_than_train_coverage": looser_than_coverage(dist_thr, reference["train"]),
        "copied_looser_than_test_known_coverage": looser_than_coverage(dist_thr, reference["test_known"])
        if selected
        else False,
        "looseness_vs_train_q95": float(dist_thr / train_q95) if train_q95 and train_q95 > 0 else float("nan"),
        "groups": groups,
        "copied_gate_rates": {
            "known_rejected_by_distance": _rate(dist_reject, test_known),
            "known_rejected_by_confidence": _rate(conf_reject, test_known),
            "unknown_accepted_by_distance": _rate(~dist_reject, test_unknown),
            "unknown_rejected_by_confidence": _rate(conf_reject, test_unknown),
        },
        "rules": rule_metrics,
    }


def _mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=float)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def _fmt(mean: float | None, std: float | None, digits: int = 2) -> str:
    if mean is None or std is None:
        return ""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean and sample standard deviation by dataset and nominal openness."""
    keys = sorted({(r["dataset"], round(float(r["nominal_openness"]), 2)) for r in rows})
    out = []
    rule_names = ("copied", "train_q95", "train_q95_distance_only", "val_known_q95", "oracle_known_q95")
    for dataset, nominal in keys:
        group = [r for r in rows if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        cell: dict[str, Any] = {
            "dataset": dataset,
            "nominal_openness": nominal,
            "actual_openness": float(np.mean([r["actual_openness_percent"] for r in group])),
            "n": len(group),
            "threshold_was_selected": bool(group[0]["threshold_was_selected"]),
        }
        for field, digits in (
            ("copied_dist", 3),
            ("looseness_vs_train_q95", 3),
        ):
            mean, std = _mean_std(r[field] for r in group)
            cell[field] = _fmt(mean, std, digits)
        for group_name in ("train", "val_known", "val_simulated", "test_known", "test_unknown"):
            mean, std = _mean_std(r["groups"][group_name]["cdf_at_copied_cut"] for r in group)
            cell[f"cdf_{group_name}"] = _fmt(mean, std, 3)
            cell[f"_cdf_{group_name}"] = mean
            mean, std = _mean_std(r["groups"][group_name]["median_dist"] for r in group)
            cell[f"median_{group_name}"] = _fmt(mean, std, 3)
            cell[f"_median_{group_name}"] = mean
        for rule in rule_names:
            for metric in ("paper_f1_percent", "FRR", "FAR"):
                mean, std = _mean_std(r["rules"][rule][metric] for r in group)
                digits = 2 if metric == "paper_f1_percent" else 3
                cell[f"{rule}_{metric}"] = _fmt(mean, std, digits)
                cell[f"_{rule}_{metric}_mean"] = mean
        out.append(cell)
    return out


def _note(cell: dict) -> str:
    """One sentence on how the two replacement cuts move this cell."""
    copied = cell["_copied_paper_f1_percent_mean"]
    train = cell["_train_q95_paper_f1_percent_mean"]
    oracle = cell["_oracle_known_q95_paper_f1_percent_mean"]
    if copied is None or train is None or oracle is None:
        return ""
    d_train = train - copied
    d_oracle = oracle - copied
    if d_train > 1.0 and d_oracle > 1.0:
        return (
            "Both replacement cuts raise micro-F1. The training-row cut is the tighter of the two, "
            "so its false-accept rate falls further and its false-reject rate rises further."
        )
    if d_oracle > 1.0 and d_train <= 1.0:
        return (
            "The test-known 0.95 cut raises micro-F1. The training-row cut is tighter than that, "
            "and the extra false rejects cancel the gain."
        )
    if d_train < -1.0:
        return (
            "The training-row cut lowers micro-F1. The copied cut is already near 95% of known test "
            "distances, and the in-sample quantile rejects known test points."
        )
    return "Micro-F1 stays within about a point of the copied cut."


def _verdict(cells: list[dict]) -> str:
    open_cells = [c for c in cells if c["threshold_was_selected"]]
    if not open_cells:
        return "No open partition was scored."
    lines = [
        "On the deployed network the copied cut accepts essentially every training row and every "
        "simulated validation unknown. Those simulated unknowns are known classes the deployed "
        "network was trained on, and their median distance matches the training rows. Known test "
        "points sit farther out, and unknown test points farther still. The confidence gate is idle: "
        "`train_q95` and `train_q95_distance_only` agree.",
    ]
    for cell in open_cells:
        lines.append(
            f"{cell['dataset']} {cell['nominal_openness']:.2f}: "
            f"copied micro-F1 {cell['copied_paper_f1_percent']} "
            f"(FRR {cell['copied_FRR']}, FAR {cell['copied_FAR']}); "
            f"training-row 0.95 cut {cell['train_q95_paper_f1_percent']} "
            f"(FRR {cell['train_q95_FRR']}, FAR {cell['train_q95_FAR']}); "
            f"test-known 0.95 cut {cell['oracle_known_q95_paper_f1_percent']} "
            f"(FRR {cell['oracle_known_q95_FRR']}, FAR {cell['oracle_known_q95_FAR']}). "
            f"The copied cut is {cell['looseness_vs_train_q95']} times the training-row quantile "
            f"and accepts {cell['cdf_test_unknown']} of unknown test points. {_note(cell)}"
        )
    return "\n\n".join(lines)


def write_reports(rows: list[dict], cells: list[dict], out_dir: Path) -> None:
    """CSV of the cell means and a markdown note next to the other result tables."""
    table_dir = RESULTS_STATIC / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "nominal_openness",
        "actual_openness",
        "n",
        "threshold_was_selected",
        "copied_dist",
        "looseness_vs_train_q95",
        "cdf_train",
        "cdf_val_known",
        "cdf_val_simulated",
        "cdf_test_known",
        "cdf_test_unknown",
        "median_train",
        "median_val_known",
        "median_val_simulated",
        "median_test_known",
        "median_test_unknown",
        "copied_paper_f1_percent",
        "copied_FRR",
        "copied_FAR",
        "train_q95_paper_f1_percent",
        "train_q95_FRR",
        "train_q95_FAR",
        "train_q95_distance_only_paper_f1_percent",
        "train_q95_distance_only_FRR",
        "train_q95_distance_only_FAR",
        "val_known_q95_paper_f1_percent",
        "val_known_q95_FRR",
        "val_known_q95_FAR",
        "oracle_known_q95_paper_f1_percent",
        "oracle_known_q95_FRR",
        "oracle_known_q95_FAR",
    ]
    csv_path = table_dir / "threshold_transfer.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for cell in cells:
            handle.write(",".join(str(cell.get(col, "")) for col in columns) + "\n")

    md_path = RESULTS_STATIC / "THRESHOLD_TRANSFER.md"
    lines = [
        "# Distance-cut transfer",
        "",
        "The copied cut is the distance threshold stored with the Phase 1 `1_NN` model.",
        "Validation chose it on the fitting-class network. The scores below are distances to the",
        "deployed network, the one trained on every known-class training row.",
        "",
        f"A coverage cut is the {COVERAGE:.2f} quantile of mean distances to the 5 nearest nodes.",
        "`train_q95` uses the deployed network's own training rows. `val_known_q95` uses the",
        "fitting-class rows in the open validation split. Both of those row sets were used to grow",
        "the deployed network, so the quantile is in-sample.",
        "`oracle_known_q95` places the same quantile on known test rows. Unknown labels are not",
        "used to choose it. It is a ceiling for this rule, not a threshold that can be selected",
        "at training time. The confidence cut stays at the saved value except in",
        "`train_q95_distance_only`, where the confidence gate is off.",
        "",
        "At openness 0 the stored cuts are the legacy defaults and rejection is off, so every rule",
        "has the same micro-F1.",
        "",
        "| Dataset | Nominal O | Copied F1 | Train q95 F1 | Train q95, distance only | Val known q95 F1 | Oracle known q95 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {copied_paper_f1_percent} | {train_q95_paper_f1_percent} | "
            "{train_q95_distance_only_paper_f1_percent} | {val_known_q95_paper_f1_percent} | "
            "{oracle_known_q95_paper_f1_percent} |".format(**cell)
        )
    lines.extend(
        [
            "",
            "| Dataset | Nominal O | Known-test CDF | Unknown-test CDF | Copied / train q95 | Copied FRR | Copied FAR | Train q95 FRR | Train q95 FAR | Oracle FRR | Oracle FAR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {cdf_test_known} | {cdf_test_unknown} | "
            "{looseness_vs_train_q95} | {copied_FRR} | {copied_FAR} | {train_q95_FRR} | "
            "{train_q95_FAR} | {oracle_known_q95_FRR} | {oracle_known_q95_FAR} |".format(**cell)
        )
    lines.extend(["", "## Reading", "", _verdict(cells), ""])
    from osr_static.paths import REPO_ROOT

    lines.append(f"Per-partition JSON is under `{out_dir.relative_to(REPO_ROOT)}`.")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run(only: dict[str, str] | None = None) -> dict[str, Any]:
    """Score every saved Phase 1 network and write the transfer report."""
    phase1 = RESULTS_STATIC / "phase1"
    records = _phase1_records(phase1)
    if not records:
        raise SystemExit(f"no Phase 1 results in {phase1}")
    out_root = RESULTS_STATIC / "threshold_transfer" / "raw"
    cache: dict[str, dict] = {}
    rows = []
    for record, path in records:
        if only:
            if "dataset" in only and record["dataset"].lower() != only["dataset"].lower():
                continue
            if "openness" in only and abs(float(record["nominal_openness"]) - float(only["openness"])) > 1e-6:
                continue
            if "partition" in only and int(record["partition_id"]) != int(only["partition"]):
                continue
        dataset = record["dataset"]
        if dataset not in cache:
            x, y = load_txt_any(dataset_file(DATA_FILES[dataset]))
            cache[dataset] = {"x": x, "y": y, "document": load_partitions(partition_file(dataset))}
        nominal = f"{float(record['nominal_openness']):.2f}"
        dest = out_root / dataset / f"p{record['partition_id']}" / f"O{nominal}.json"
        print(
            f"{dataset} p{record['partition_id']} O{nominal} nodes={record.get('n_nodes')}",
            flush=True,
        )
        payload = analyze_record(record, path, cache[dataset])
        atomic_json(dest, payload)
        rows.append(payload)
    cells = aggregate(rows)
    write_reports(rows, cells, out_root)
    return {"n": len(rows), "cells": cells, "verdict": _verdict(cells)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate the 1_NN distance-cut transfer hypothesis.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--openness", default=None)
    parser.add_argument("--partition", default=None)
    args = parser.parse_args(argv)
    only = {}
    if args.dataset:
        only["dataset"] = args.dataset
    if args.openness:
        only["openness"] = args.openness
    if args.partition:
        only["partition"] = args.partition
    result = run(only or None)
    print(result["verdict"])
    print(f"partitions: {result['n']}")


if __name__ == "__main__":
    main()
