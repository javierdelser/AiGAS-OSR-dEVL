"""Per-class Mahalanobis k-NN on a saved Growing Neural Gas. No gas is grown.

Each class keeps the Ledoit-Wolf covariance of its relabeled nodes. A query's
distance to a node of that class is the Mahalanobis distance under that
covariance, so the neighbors are not the Euclidean ones. The reject score is
the mean distance to the k neighbors of the chosen class. ``maha_k5`` and
``maha_k1`` also choose the class by that distance. ``maha_k5_eucl_label``
keeps the Euclidean 5-NN label and only replaces the score.

``val`` maximises paper micro-F1 on the open validation split. ``train_q95``
is the 0.95 quantile of the score on the known training rows. Oracle cuts use
the test labels and are stored beside the deployable cut.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

from osr_static.data import LabelVault
from osr_static.gng_variants import _pack_rule, _select_val
from osr_static.improvements import (
    N_NEIGHBORS,
    _fmt,
    _load_bundle,
    _matches,
    _mean_std,
    _phase1_records,
    _scale,
    closed_labels,
    knn_predict,
    relabel_from_nearest,
    to_numeric,
)
from osr_static.jobs import atomic_json, experiment_root
from osr_static.paths import RESULTS_STATIC
from osr_static.metrics import normalize_label
from osr_static.partitions import find_record, materialize
from osr_static.scores import linear_quantile

COVERAGE = 0.95
MIN_ROWS = 8


def whiten_matrix(precision: np.ndarray) -> np.ndarray:
    """``W`` such that ``||W.T (x - y)||`` is the Mahalanobis distance of ``precision``."""
    precision = np.asarray(precision, dtype=float)
    precision = 0.5 * (precision + precision.T)
    evals, evecs = np.linalg.eigh(precision)
    evals = np.maximum(evals, 1e-8)
    return evecs * np.sqrt(evals)


def fit_precision(points: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf precision. Falls back to a scaled identity if the fit is singular."""
    points = np.asarray(points, dtype=float)
    width = points.shape[1]
    if points.shape[0] < 2:
        return np.eye(width)
    try:
        precision = np.asarray(LedoitWolf().fit(points).precision_, dtype=float)
    except (ValueError, np.linalg.LinAlgError):
        variance = np.var(points, axis=0)
        variance = np.maximum(variance, 1e-6)
        precision = np.diag(1.0 / variance)
    if not np.isfinite(precision).all():
        return np.eye(width)
    return 0.5 * (precision + precision.T)


class ClassMetrics:
    """One whitened neighbor index per class. A tiny class uses the pooled covariance."""

    def __init__(self, nodes: np.ndarray, labels: np.ndarray) -> None:
        nodes = np.asarray(nodes, dtype=float)
        labels = np.asarray(labels)
        pooled = whiten_matrix(fit_precision(nodes))
        self._models: dict[Any, tuple[np.ndarray, NearestNeighbors]] = {}
        for label in sorted(set(labels.tolist()), key=lambda value: (str(type(value)), str(value))):
            subset = nodes[labels == label]
            if subset.shape[0] == 0:
                continue
            transform = whiten_matrix(fit_precision(subset)) if subset.shape[0] >= MIN_ROWS else pooled
            whitened = subset @ transform
            neighbors = min(whitened.shape[0], max(N_NEIGHBORS, 1))
            algorithm = "ball_tree" if whitened.shape[0] >= 20 else "brute"
            model = NearestNeighbors(n_neighbors=neighbors, algorithm=algorithm)
            model.fit(whitened)
            self._models[label] = (transform, model)

    def labels(self) -> list[Any]:
        return list(self._models)

    def mean_distance(self, queries: np.ndarray, label: Any, k: int) -> np.ndarray:
        transform, model = self._models[label]
        k_use = min(int(k), model.n_samples_fit_)
        distances, _ = model.kneighbors(np.asarray(queries, dtype=float) @ transform, n_neighbors=k_use)
        return distances.mean(axis=1)

    def predict(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Winning-class mean distance and the integer label of that class."""
        queries = np.asarray(queries, dtype=float)
        labels = self.labels()
        if not labels:
            raise ValueError("no class has nodes")
        table = np.column_stack([self.mean_distance(queries, label, k) for label in labels])
        chosen = np.argmin(table, axis=1)
        score = table[np.arange(queries.shape[0]), chosen]
        pred = np.asarray(labels, dtype=int)[chosen]
        return score, pred

    def score_known_label(self, queries: np.ndarray, predicted: np.ndarray, k: int) -> np.ndarray:
        """Mean distance to class ``predicted``. An unseen label uses the nearest stored class."""
        queries = np.asarray(queries, dtype=float)
        predicted = np.asarray(predicted)
        out = np.empty(queries.shape[0], dtype=float)
        known = set(self._models)
        for label in np.unique(predicted):
            mask = predicted == label
            use = label if label in known else self.labels()[0]
            out[mask] = self.mean_distance(queries[mask], use, k)
        return out


def _cut_pair(score_te, pred_te, score_tr, score_val, pred_val, y_val, y_true, known, fitting, openness, vault, stem):
    tau_val = _select_val(score_val, pred_val, y_val, fitting, openness)
    tau_tr = None if openness <= 0 else float(linear_quantile(score_tr, COVERAGE))
    return {
        f"{stem}__val": _pack_rule(stem, score_te, pred_te, y_true, known, openness, tau_val, vault),
        f"{stem}__train_q95": _pack_rule(stem, score_te, pred_te, y_true, known, openness, tau_tr, vault),
    }


def analyze_record(record: dict, json_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    """One saved network. Covariance is estimated from its nodes only."""
    dataset = record["dataset"]
    nominal = float(record["nominal_openness"])
    arrays = materialize(
        bundle["x"],
        bundle["y"],
        find_record(bundle["document"], record["partition_id"], nominal),
    )
    with np.load(json_path.with_suffix(".npz"), allow_pickle=True) as saved:
        nodes = np.asarray(saved["node_positions"], dtype=float)
        mean = np.asarray(saved["scaler_mean"], dtype=float)
        scale = np.asarray(saved["scaler_scale"], dtype=float)
        y_true = np.array([normalize_label(v) for v in saved["y_true"]], dtype=object)
    openness = float(record["actual_openness_percent"])
    known = [str(v) for v in record["known_classes"]]
    fitting = [str(v) for v in record["fitting_classes"]]
    nodes_s = _scale(nodes, mean, scale)
    x_tr = _scale(arrays["X_tr"], mean, scale)
    x_te = _scale(arrays["X_te"], mean, scale)
    x_val = _scale(arrays["X_val_open"], mean, scale)
    y_tr_num = to_numeric(dataset, arrays["y_tr"])
    node_num = relabel_from_nearest(nodes_s, x_tr, y_tr_num)
    vault = None if openness <= 0 else LabelVault(y_true, known)
    metrics = ClassMetrics(nodes_s, node_num)

    rules: dict[str, Any] = {}
    for k, stem in ((N_NEIGHBORS, "maha_k5"), (1, "maha_k1")):
        score_te, pred_te = metrics.predict(x_te, k)
        score_tr, _ = metrics.predict(x_tr, k)
        score_val, pred_val = metrics.predict(x_val, k)
        rules.update(
            _cut_pair(
                score_te,
                closed_labels(dataset, pred_te),
                score_tr,
                score_val,
                closed_labels(dataset, pred_val),
                np.array([normalize_label(v) for v in arrays["y_val_open"]], dtype=object),
                y_true,
                known,
                fitting,
                openness,
                vault,
                stem,
            )
        )

    y_val = np.array([normalize_label(v) for v in arrays["y_val_open"]], dtype=object)
    _, _, pred_eucl_te = knn_predict(nodes_s, node_num, x_te, N_NEIGHBORS)
    _, _, pred_eucl_tr = knn_predict(nodes_s, node_num, x_tr, N_NEIGHBORS)
    _, _, pred_eucl_val = knn_predict(nodes_s, node_num, x_val, N_NEIGHBORS)
    rules.update(
        _cut_pair(
            metrics.score_known_label(x_te, pred_eucl_te, N_NEIGHBORS),
            closed_labels(dataset, pred_eucl_te),
            metrics.score_known_label(x_tr, pred_eucl_tr, N_NEIGHBORS),
            metrics.score_known_label(x_val, pred_eucl_val, N_NEIGHBORS),
            closed_labels(dataset, pred_eucl_val),
            y_val,
            y_true,
            known,
            fitting,
            openness,
            vault,
            "maha_k5_eucl_label",
        )
    )
    return {
        "ok": True,
        "dataset": dataset,
        "partition_id": int(record["partition_id"]),
        "nominal_openness": nominal,
        "actual_openness_percent": openness,
        "n_nodes": int(nodes_s.shape[0]),
        "n_classes": len(metrics.labels()),
        "rules": rules,
    }


RULES = [
    "maha_k5__val",
    "maha_k5__train_q95",
    "maha_k1__val",
    "maha_k1__train_q95",
    "maha_k5_eucl_label__val",
    "maha_k5_eucl_label__train_q95",
]


def write_report(rows: list[dict]) -> None:
    keys = sorted({(r["dataset"], round(float(r["nominal_openness"]), 2)) for r in rows})
    lines = [
        "# Per-class Mahalanobis k-NN",
        "",
        "No gas is grown. Nodes are the saved Phase 1 network, relabeled from the",
        "nearest training row. Each class uses the Ledoit-Wolf covariance of its",
        "nodes. The score is the mean Mahalanobis distance to the k neighbors of",
        "the chosen class. `k5` and `k1` choose the class by that distance.",
        "`eucl label` keeps the Euclidean 5-NN class and only replaces the score.",
        "`val` maximises paper micro-F1 on the open validation split.",
        "`q95` is the 0.95 quantile of the score on the known training rows.",
        "",
        "| Dataset | O | n | k5 val | k5 q95 | k1 val | k1 q95 | k5, Euclidean label, val | k5, Euclidean label, q95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cells = []
    for dataset, nominal in keys:
        group = [r for r in rows if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        cell: dict[str, Any] = {"dataset": dataset, "nominal_openness": nominal, "n": len(group)}
        for rule in RULES:
            for metric, digits in (("paper_f1_percent", 2), ("FRR", 3), ("FAR", 3), ("closed_acc", 3), ("auroc", 3)):
                mean, std = _mean_std(r["rules"].get(rule, {}).get(metric) for r in group)
                cell[f"{rule}_{metric}"] = _fmt(mean, std, digits)
        cells.append(cell)
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {n} | {maha_k5__val_paper_f1_percent} | "
            "{maha_k5__train_q95_paper_f1_percent} | {maha_k1__val_paper_f1_percent} | "
            "{maha_k1__train_q95_paper_f1_percent} | {maha_k5_eucl_label__val_paper_f1_percent} | "
            "{maha_k5_eucl_label__train_q95_paper_f1_percent} |".format(**cell)
        )
    lines.extend(
        [
            "",
            "| Dataset | O | k5 val FRR | k5 val FAR | k5 q95 FRR | k5 q95 FAR | Euclidean-label val FRR | Euclidean-label val FAR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {maha_k5__val_FRR} | {maha_k5__val_FAR} | "
            "{maha_k5__train_q95_FRR} | {maha_k5__train_q95_FAR} | "
            "{maha_k5_eucl_label__val_FRR} | {maha_k5_eucl_label__val_FAR} |".format(**cell)
        )
    lines.append("")
    path = RESULTS_STATIC / "GNG_MAHALANOBIS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    table = RESULTS_STATIC / "tables" / "gng_mahalanobis.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "nominal_openness", "n"]
    for rule in RULES:
        for metric in ("paper_f1_percent", "FRR", "FAR", "closed_acc", "auroc"):
            columns.append(f"{rule}_{metric}")
    with table.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for cell in cells:
            handle.write(",".join(str(cell.get(col, "")) for col in columns) + "\n")


def score_one(json_path: str) -> dict[str, Any]:
    path = Path(json_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    nominal = f"{float(record['nominal_openness']):.2f}"
    dest = (
        RESULTS_STATIC
        / "gng_mahalanobis"
        / "raw"
        / record["dataset"]
        / f"p{record['partition_id']}"
        / f"O{nominal}.json"
    )
    if dest.is_file():
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if existing.get("ok") and "maha_k5_eucl_label__val" in existing.get("rules", {}):
            return existing
    bundle = _load_bundle(record["dataset"])
    payload = analyze_record(record, path, bundle)
    atomic_json(dest, payload)
    return payload


def run_gng_mahalanobis(only: dict[str, str] | None = None, jobs: int = 1, resume: bool = True) -> dict[str, Any]:
    """Score every saved Phase 1 network and write ``results/GNG_MAHALANOBIS.md``."""
    del resume
    records = _phase1_records(RESULTS_STATIC / "phase1")
    selected = [path for record, path in records if _matches(record, only)]
    if not selected:
        raise SystemExit("no Phase 1 models matched")
    rows = []
    workers = max(1, int(jobs))
    print(f"{len(selected)} Mahalanobis jobs, {workers} workers", flush=True)
    if workers == 1:
        for path in selected:
            print(f"score {path.parent.parent.name} {path.parent.name}", flush=True)
            rows.append(score_one(str(path)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(score_one, str(path)): path for path in selected}
            for future in as_completed(futures):
                path = futures[future]
                print(f"done {path.parent.parent.name} {path.parent.name}", flush=True)
                rows.append(future.result())
    rows.sort(key=lambda row: (row["dataset"], float(row["nominal_openness"]), int(row["partition_id"])))
    write_report(rows)
    return {"n": len(rows), "report": str(RESULTS_STATIC / "GNG_MAHALANOBIS.md")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Per-class Mahalanobis k-NN on saved GNG nodes. No gas is trained.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--openness", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    only = {}
    if args.dataset:
        only["dataset"] = args.dataset
    if args.openness:
        only["openness"] = args.openness
    if args.partition:
        only["partition"] = args.partition
    print(json.dumps(run_gng_mahalanobis(only or None, jobs=args.jobs, resume=args.resume)), flush=True)


if __name__ == "__main__":
    main()
