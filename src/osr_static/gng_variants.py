"""Scores and classifiers that use a saved Growing Neural Gas as the only model.

No gas is grown. Every rule starts from the Phase 1 nodes. Nodes are relabeled
from the nearest training row, then a classifier is fit on those nodes and a
reject score is cut on training rows (0.95 quantile) or on open validation
(paper micro-F1, no 95% coverage floor). Oracle cuts are stored and marked.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC, SVC

from osr_static.baselines.osnn import scores as osnn_scores
from osr_static.data import LabelVault
from osr_static.improvements import (
    N_NEIGHBORS,
    _fmt,
    _load_bundle,
    _matches,
    _mean_std,
    _metrics,
    _phase1_records,
    _scale,
    closed_labels,
    knn_predict,
    local_spacing,
    relabel_from_nearest,
    to_numeric,
)
from osr_static.jobs import atomic_json, experiment_root
from osr_static.paths import RESULTS_STATIC
from osr_static.metrics import apply_threshold, auroc_unknown, normalize_label
from osr_static.oracle import oracle_tau
from osr_static.partitions import find_record, materialize
from osr_static.scores import linear_quantile
from osr_static.selection import select_threshold

COVERAGE = 0.95
RBF_NODE_CAP = 8000
OSNN_GRID = np.linspace(0.5, 1.0, 51)


def class_conditional_distance(
    nodes: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    predicted: np.ndarray,
    k: int = N_NEIGHBORS,
) -> np.ndarray:
    """Mean distance to ``k`` nodes of the predicted class. Higher is more unknown."""
    nodes = np.asarray(nodes, dtype=float)
    labels = np.asarray(labels)
    queries = np.asarray(queries, dtype=float)
    predicted = np.asarray(predicted)
    out = np.full(queries.shape[0], np.inf, dtype=float)
    for label in np.unique(predicted):
        mask = predicted == label
        subset = nodes[labels == label]
        if subset.shape[0] == 0:
            continue
        k_use = min(int(k), subset.shape[0])
        model = NearestNeighbors(n_neighbors=k_use, algorithm="ball_tree")
        model.fit(subset)
        distances, _ = model.kneighbors(queries[mask])
        out[mask] = distances.mean(axis=1)
    return out


def class_spacing_scores(
    nodes: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """Nearest-node distance divided by the mean spacing of that class's nodes."""
    nodes = np.asarray(nodes, dtype=float)
    labels = np.asarray(labels)
    spacing = local_spacing(nodes)
    scale = {}
    for label in np.unique(labels):
        members = spacing[labels == label]
        scale[label] = float(members.mean()) if members.size else 1.0
    model = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    model.fit(nodes)
    distances, _ = model.kneighbors(np.asarray(queries, dtype=float))
    denom = np.array([max(scale.get(label, 1.0), 1e-12) for label in np.asarray(predicted)])
    return distances.ravel() / denom


def _margin_score(decision: np.ndarray) -> np.ndarray:
    """Higher when the winning class is less confident."""
    decision = np.asarray(decision, dtype=float)
    if decision.ndim == 1:
        winning = np.abs(decision)
    else:
        winning = np.max(decision, axis=1)
    return -winning


def _subsample(nodes: np.ndarray, labels: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if nodes.shape[0] <= cap:
        return nodes, labels
    rng = np.random.default_rng(int(seed))
    index = rng.choice(nodes.shape[0], size=cap, replace=False)
    return nodes[index], labels[index]


def _predict_margin(model, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(queries)
    return _margin_score(model.decision_function(queries)), pred


def _fit_linear_svm(nodes: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels).astype(str)
    model = LinearSVC(C=1.0, dual="auto", max_iter=4000)
    model.fit(nodes, labels)
    return model


def _fit_rbf_svm(nodes: np.ndarray, labels: np.ndarray, seed: int):
    nodes, labels = _subsample(nodes, labels, RBF_NODE_CAP, seed)
    labels = np.asarray(labels).astype(str)
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)
    model.fit(nodes, labels)
    return model


def _fit_logreg(nodes: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels).astype(str)
    model = LogisticRegression(max_iter=400, solver="lbfgs")
    model.fit(nodes, labels)
    return model


def _predict_proba(model, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(queries)
    pred = model.classes_[np.argmax(proba, axis=1)]
    return 1.0 - proba.max(axis=1), pred


def _select_val(scores, pred, y_val, known, openness: float) -> float | None:
    if openness <= 0:
        return None
    tau, _ = select_threshold(scores, pred, y_val, known)
    return float(tau)


def _select_osnn_grid(scores, pred, y_val, known, openness: float) -> float | None:
    if openness <= 0:
        return None
    best = (-1.0, 0.75)
    for threshold in OSNN_GRID:
        guessed = apply_threshold(scores, pred, float(threshold))
        from osr_static.metrics import micro_f1_reference

        f1 = micro_f1_reference(y_val, guessed, known)
        closer = abs(float(threshold) - 0.75) < abs(best[1] - 0.75)
        if f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and closer):
            best = (float(f1), float(threshold))
    return best[1]


def _pack_rule(
    name: str,
    scores_te: np.ndarray,
    pred_te: np.ndarray,
    y_true: np.ndarray,
    known: list[str],
    openness: float,
    tau: float | None,
    vault: LabelVault | None,
) -> dict[str, Any]:
    if tau is None or openness <= 0:
        guessed = pred_te
    else:
        guessed = apply_threshold(scores_te, pred_te, tau)
    payload = _metrics(y_true, guessed, pred_te, known)
    payload["tau"] = None if tau is None else float(tau)
    unknown = ~np.isin(y_true, [normalize_label(v) for v in known])
    payload["auroc"] = None if openness <= 0 else auroc_unknown(scores_te, unknown)
    if vault is not None and openness > 0:
        oracle = oracle_tau(vault, scores_te, pred_te)
        payload["oracle_f1_percent"] = oracle["f1_percent"]
        payload["oracle_tau"] = oracle["tau"]
        payload["regret"] = float(oracle["f1_percent"] - payload["paper_f1_percent"])
    return payload


def analyze_record(record: dict, json_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    """One saved network. Classifiers and scores are fit on its nodes only."""
    dataset = record["dataset"]
    arrays = materialize(
        bundle["x"],
        bundle["y"],
        find_record(bundle["document"], record["partition_id"], record["nominal_openness"]),
    )
    with np.load(json_path.with_suffix(".npz"), allow_pickle=True) as saved:
        nodes = np.asarray(saved["node_positions"], dtype=float)
        mean = np.asarray(saved["scaler_mean"], dtype=float)
        scale = np.asarray(saved["scaler_scale"], dtype=float)
        y_true = np.array([normalize_label(v) for v in saved["y_true"]], dtype=object)

    openness = float(record["actual_openness_percent"])
    known = [str(v) for v in record["known_classes"]]
    fitting = [str(v) for v in record["fitting_classes"]]
    seed = int(record["seed"])

    nodes_s = _scale(nodes, mean, scale)
    x_tr = _scale(arrays["X_tr"], mean, scale)
    x_te = _scale(arrays["X_te"], mean, scale)
    x_val = _scale(arrays["X_val_open"], mean, scale)
    y_tr_num = to_numeric(dataset, arrays["y_tr"])
    y_val = np.array([normalize_label(v) for v in arrays["y_val_open"]], dtype=object)
    node_num = relabel_from_nearest(nodes_s, x_tr, y_tr_num)
    node_str = closed_labels(dataset, node_num)
    vault = None if openness <= 0 else LabelVault(y_true, known)

    dist5_te, conf5_te, pred5_num = knn_predict(nodes_s, node_num, x_te, N_NEIGHBORS)
    dist5_tr, conf5_tr, _ = knn_predict(nodes_s, node_num, x_tr, N_NEIGHBORS)
    dist5_val, conf5_val, pred5_val_num = knn_predict(nodes_s, node_num, x_val, N_NEIGHBORS)
    pred5 = closed_labels(dataset, pred5_num)
    pred5_val = closed_labels(dataset, pred5_val_num)

    dist1_te, _, pred1_num = knn_predict(nodes_s, node_num, x_te, 1)
    dist1_tr, _, _ = knn_predict(nodes_s, node_num, x_tr, 1)
    dist1_val, _, pred1_val_num = knn_predict(nodes_s, node_num, x_val, 1)
    pred1 = closed_labels(dataset, pred1_num)
    pred1_val = closed_labels(dataset, pred1_val_num)

    svm_lin = _fit_linear_svm(nodes_s, node_str)
    svm_lin_te, svm_lin_pred_te = _predict_margin(svm_lin, x_te)
    svm_lin_tr, _ = _predict_margin(svm_lin, x_tr)
    svm_lin_val, svm_lin_pred_val = _predict_margin(svm_lin, x_val)
    svm_lin_pred_te = np.array([normalize_label(v) for v in svm_lin_pred_te], dtype=object)
    svm_lin_pred_val = np.array([normalize_label(v) for v in svm_lin_pred_val], dtype=object)

    svm_rbf = _fit_rbf_svm(nodes_s, node_str, seed)
    svm_rbf_te, svm_rbf_pred_te = _predict_margin(svm_rbf, x_te)
    svm_rbf_tr, _ = _predict_margin(svm_rbf, x_tr)
    svm_rbf_val, svm_rbf_pred_val = _predict_margin(svm_rbf, x_val)
    svm_rbf_pred_te = np.array([normalize_label(v) for v in svm_rbf_pred_te], dtype=object)
    svm_rbf_pred_val = np.array([normalize_label(v) for v in svm_rbf_pred_val], dtype=object)

    log_model = _fit_logreg(nodes_s, node_str)
    log_te, log_pred_te = _predict_proba(log_model, x_te)
    log_tr, _ = _predict_proba(log_model, x_tr)
    log_val, log_pred_val = _predict_proba(log_model, x_val)
    log_pred_te = np.array([normalize_label(v) for v in log_pred_te], dtype=object)
    log_pred_val = np.array([normalize_label(v) for v in log_pred_val], dtype=object)

    ratio_te, ratio_pred_te = osnn_scores(nodes_s, node_str, x_te)
    ratio_tr, _ = osnn_scores(nodes_s, node_str, x_tr)
    ratio_val, ratio_pred_val = osnn_scores(nodes_s, node_str, x_val)
    ratio_pred_te = np.array([normalize_label(v) for v in ratio_pred_te], dtype=object)
    ratio_pred_val = np.array([normalize_label(v) for v in ratio_pred_val], dtype=object)

    class_dist_te = class_conditional_distance(nodes_s, node_num, x_te, pred5_num)
    class_dist_tr = class_conditional_distance(nodes_s, node_num, x_tr, to_numeric(dataset, arrays["y_tr"]))
    class_dist_val = class_conditional_distance(nodes_s, node_num, x_val, pred5_val_num)

    class_sp_te = class_spacing_scores(nodes_s, node_num, x_te, pred5_num)
    class_sp_tr = class_spacing_scores(nodes_s, node_num, x_tr, y_tr_num)
    class_sp_val = class_spacing_scores(nodes_s, node_num, x_val, pred5_val_num)

    conf_te = 1.0 - conf5_te
    conf_tr = 1.0 - conf5_tr
    conf_val = 1.0 - conf5_val

    families = [
        ("knn5__dist5", dist5_te, pred5, dist5_tr, dist5_val, pred5_val, _select_val),
        ("knn1__dist1", dist1_te, pred1, dist1_tr, dist1_val, pred1_val, _select_val),
        ("knn5__conf", conf_te, pred5, conf_tr, conf_val, pred5_val, _select_val),
        ("knn5__class_dist", class_dist_te, pred5, class_dist_tr, class_dist_val, pred5_val, _select_val),
        ("knn5__class_spacing", class_sp_te, pred5, class_sp_tr, class_sp_val, pred5_val, _select_val),
        ("svm_linear__margin", svm_lin_te, svm_lin_pred_te, svm_lin_tr, svm_lin_val, svm_lin_pred_val, _select_val),
        ("svm_rbf__margin", svm_rbf_te, svm_rbf_pred_te, svm_rbf_tr, svm_rbf_val, svm_rbf_pred_val, _select_val),
        ("logreg__proba", log_te, log_pred_te, log_tr, log_val, log_pred_val, _select_val),
        ("osnn__ratio", ratio_te, ratio_pred_te, ratio_tr, ratio_val, ratio_pred_val, _select_osnn_grid),
    ]

    rules: dict[str, Any] = {}
    for stem, score_te, pred_te, score_tr, score_val, pred_val, selector in families:
        tau_val = selector(score_val, pred_val, y_val, fitting, openness)
        tau_tr = None if openness <= 0 else float(linear_quantile(score_tr, COVERAGE))
        rules[f"{stem}__val"] = _pack_rule(stem, score_te, pred_te, y_true, known, openness, tau_val, vault)
        rules[f"{stem}__train_q95"] = _pack_rule(stem, score_te, pred_te, y_true, known, openness, tau_tr, vault)
        if openness > 0:
            oracle = oracle_tau(vault, score_te, pred_te)
            rules[f"{stem}__ORACLE"] = {
                "paper_f1_percent": oracle["f1_percent"],
                "FRR": None,
                "FAR": None,
                "closed_acc": rules[f"{stem}__val"]["closed_acc"],
                "tau": oracle["tau"],
                "auroc": rules[f"{stem}__val"]["auroc"],
            }

    return {
        "ok": True,
        "dataset": dataset,
        "partition_id": int(record["partition_id"]),
        "nominal_openness": float(record["nominal_openness"]),
        "actual_openness_percent": openness,
        "n_nodes": int(nodes_s.shape[0]),
        "rbf_node_cap": RBF_NODE_CAP,
        "rules": rules,
    }


PRIMARY = [
    "knn5__dist5__val",
    "knn5__dist5__train_q95",
    "knn1__dist1__val",
    "knn1__dist1__train_q95",
    "svm_linear__margin__val",
    "svm_linear__margin__train_q95",
    "svm_rbf__margin__val",
    "svm_rbf__margin__train_q95",
    "logreg__proba__val",
    "logreg__proba__train_q95",
    "osnn__ratio__val",
    "osnn__ratio__train_q95",
]
SCORES = [
    "knn5__conf__val",
    "knn5__conf__train_q95",
    "knn5__class_dist__val",
    "knn5__class_dist__train_q95",
    "knn5__class_spacing__val",
    "knn5__class_spacing__train_q95",
]


def write_report(rows: list[dict]) -> None:
    keys = sorted({(r["dataset"], round(float(r["nominal_openness"]), 2)) for r in rows})
    lines = [
        "# GNG scores and classifiers",
        "",
        "The gas is the saved Phase 1 network. Nodes are relabeled from the nearest",
        "training row. Classifiers are fit on those nodes. `val` maximises paper",
        "micro-F1 on the open validation split with no 95% coverage floor.",
        "`train_q95` is the 0.95 quantile of the score on the known training rows.",
        "RBF SVM draws at most 8000 nodes. Linear SVM, logistic regression, k-NN,",
        "and OSNN use every node. `ORACLE` rows use the test labels.",
        "",
        "| Dataset | O | n | kNN-5 dist val | kNN-5 dist q95 | kNN-1 dist val | Linear SVM val | RBF SVM val | LogReg val | OSNN-on-nodes val |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cells = []
    for dataset, nominal in keys:
        group = [r for r in rows if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        cell: dict[str, Any] = {"dataset": dataset, "nominal_openness": nominal, "n": len(group)}
        for rule in PRIMARY + SCORES:
            for metric, digits in (("paper_f1_percent", 2), ("FRR", 3), ("FAR", 3), ("closed_acc", 3), ("auroc", 3)):
                mean, std = _mean_std(r["rules"].get(rule, {}).get(metric) for r in group)
                cell[f"{rule}_{metric}"] = _fmt(mean, std, digits)
        cells.append(cell)
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {n} | {knn5__dist5__val_paper_f1_percent} | "
            "{knn5__dist5__train_q95_paper_f1_percent} | {knn1__dist1__val_paper_f1_percent} | "
            "{svm_linear__margin__val_paper_f1_percent} | {svm_rbf__margin__val_paper_f1_percent} | "
            "{logreg__proba__val_paper_f1_percent} | {osnn__ratio__val_paper_f1_percent} |".format(**cell)
        )
    lines.extend(
        [
            "",
            "| Dataset | O | kNN-5 conf val | Class-cond. dist val | Class spacing val | Class-cond. dist q95 | Class spacing q95 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {knn5__conf__val_paper_f1_percent} | "
            "{knn5__class_dist__val_paper_f1_percent} | {knn5__class_spacing__val_paper_f1_percent} | "
            "{knn5__class_dist__train_q95_paper_f1_percent} | "
            "{knn5__class_spacing__train_q95_paper_f1_percent} |".format(**cell)
        )
    lines.append("")
    path = RESULTS_STATIC / "GNG_VARIANTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    table = RESULTS_STATIC / "tables" / "gng_variants.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "nominal_openness", "n"]
    for rule in PRIMARY + SCORES:
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
        / "gng_variants"
        / "raw"
        / record["dataset"]
        / f"p{record['partition_id']}"
        / f"O{nominal}.json"
    )
    if dest.is_file():
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if existing.get("ok") and "osnn__ratio__val" in existing.get("rules", {}):
            return existing
    bundle = _load_bundle(record["dataset"])
    payload = analyze_record(record, path, bundle)
    atomic_json(dest, payload)
    return payload


def run_gng_variants(only: dict[str, str] | None = None, jobs: int = 1, resume: bool = True) -> dict[str, Any]:
    """Score every saved Phase 1 network and write ``results/GNG_VARIANTS.md``."""
    del resume
    records = _phase1_records(RESULTS_STATIC / "phase1")
    selected = [path for record, path in records if _matches(record, only)]
    if not selected:
        raise SystemExit("no Phase 1 models matched")
    rows = []
    workers = max(1, int(jobs))
    print(f"{len(selected)} GNG-variant jobs, {workers} workers", flush=True)
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
    return {"n": len(rows), "report": str(RESULTS_STATIC / "GNG_VARIANTS.md")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scores and classifiers on saved GNG nodes. No gas is trained.")
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
    print(json.dumps(run_gng_variants(only or None, jobs=args.jobs, resume=args.resume)), flush=True)


if __name__ == "__main__":
    main()
