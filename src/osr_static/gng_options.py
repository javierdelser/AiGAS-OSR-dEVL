"""Six rescores of a saved Growing Neural Gas. No gas is grown.

1. ``pick_cut`` scores three cuts of the relabeled 5-NN distance on the open
   validation set and deploys the one with the higher paper micro-F1: the
   validation sweep, the training-row 0.95 quantile, and the Phase 1 copied
   distance threshold. The copied candidate is that distance threshold alone.
2. ``svm_label__gng_val`` takes the known-class label from the raw-feature
   RBF SVM already selected for the partition, and rejects on the 5-NN
   distance. The distance cut is chosen on validation with those labels.
3. ``landmark`` fits a logistic regression on the distance from the query to
   the nearest node of each class. ``proba`` rejects on one minus the largest
   class probability. ``min`` rejects on the smallest of those distances.
4. Distance-weighted k-NN for k in {3, 7, 11}. ``mean`` is the mean neighbor
   distance. ``kth`` is the distance to the k-th neighbor.
5. ``per_class__dist5`` gives each class its own distance quantile and chooses
   the quantile on validation. A class with fewer than five known validation
   rows uses its training rows.
6. ``either_rejects`` rejects when the GNG gate or the SVM probability gate
   rejects. ``both_rejects`` rejects only when both do. The label is the SVM
   label. Both gates are chosen on validation.

Oracle numbers use the test labels and are stored beside the deployable cut.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from osr_static.data import LabelVault
from osr_static.gng_variants import _pack_rule, _select_val
from osr_static.improvements import (
    N_NEIGHBORS,
    _baseline_params,
    _fmt,
    _load_bundle,
    _matches,
    _mean_std,
    _metrics,
    _phase1_records,
    _scale,
    closed_labels,
    knn_predict,
    relabel_from_nearest,
    to_numeric,
)
from osr_static.jobs import atomic_json, experiment_root
from osr_static.paths import RESULTS_STATIC
from osr_static.metrics import apply_threshold, micro_f1_reference, normalize_label
from osr_static.partitions import find_record, materialize
from osr_static.scores import linear_quantile
from osr_static.selection import Q_GRID, select_q

COVERAGE = 0.95
MIN_CLASS_ROWS = 5
KS = (3, 7, 11)
CUT_ORDER = ("val", "train_q95", "copied")


def choose_cut(val_f1: dict[str, float], taus: dict[str, float], scores_val: np.ndarray) -> str:
    """Name of the cut with the higher validation F1.

    A tie keeps the threshold closer to the median finite validation score,
    then the earlier name in ``CUT_ORDER``.
    """
    finite = np.asarray(scores_val, dtype=float)
    finite = finite[np.isfinite(finite)]
    center = float(np.median(finite)) if finite.size else 0.0
    best = max(float(val_f1[name]) for name in CUT_ORDER)
    tied = [name for name in CUT_ORDER if abs(float(val_f1[name]) - best) <= 1e-12]
    tied.sort(key=lambda name: (abs(float(taus[name]) - center), CUT_ORDER.index(name)))
    return tied[0]


def class_landmarks(
    nodes: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    class_order: Sequence[object],
) -> np.ndarray:
    """Distance from each query to the nearest node of each class, in ``class_order``."""
    nodes = np.asarray(nodes, dtype=float)
    labels = np.asarray(labels)
    queries = np.asarray(queries, dtype=float)
    out = np.full((queries.shape[0], len(class_order)), np.inf, dtype=float)
    for column, label in enumerate(class_order):
        subset = nodes[labels == label]
        if subset.shape[0] == 0:
            continue
        model = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        model.fit(subset)
        distances, _ = model.kneighbors(queries)
        out[:, column] = distances.ravel()
    return out


def finite_landmarks(raw: np.ndarray) -> np.ndarray:
    """Replace a missing class distance with the largest finite distance in the matrix."""
    out = np.array(raw, dtype=float, copy=True)
    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros_like(out)
    cap = float(np.max(out[finite]))
    out[~finite] = cap
    return out


def knn_distance_weighted(
    nodes: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance-weighted class, mean neighbor distance, and distance to the k-th neighbor."""
    nodes = np.asarray(nodes, dtype=float)
    queries = np.asarray(queries, dtype=float)
    k_use = min(int(k), nodes.shape[0])
    model = KNeighborsClassifier(n_neighbors=k_use, weights="distance", algorithm="ball_tree")
    model.fit(nodes, np.asarray(labels).ravel())
    distances, _ = model.kneighbors(queries)
    pred = np.rint(model.predict(queries).astype(float)).astype(int)
    return distances.mean(axis=1), distances[:, -1], pred


def class_quantiles(scores: np.ndarray, labels: Sequence[object], classes: Sequence[object], q: float) -> dict[str, float]:
    """Quantile of ``scores`` inside each class that has at least one row."""
    scores = np.asarray(scores, dtype=float)
    labels = np.array([normalize_label(v) for v in np.asarray(labels).ravel()], dtype=object)
    found = {}
    for label in classes:
        key = normalize_label(label)
        member = scores[labels == key]
        member = member[np.isfinite(member)]
        if member.size == 0:
            continue
        found[key] = float(linear_quantile(member, q))
    return found


def class_cut_table(
    scores_val: np.ndarray,
    y_val: Sequence[object],
    scores_train: np.ndarray,
    y_train: Sequence[object],
    classes: Sequence[object],
    q: float,
    min_rows: int = MIN_CLASS_ROWS,
) -> tuple[dict[str, float], float]:
    """Per-class quantile. Validation rows win when a class has ``min_rows`` of them."""
    scores_val = np.asarray(scores_val, dtype=float)
    y_val_n = np.array([normalize_label(v) for v in np.asarray(y_val).ravel()], dtype=object)
    from_val = class_quantiles(scores_val, y_val_n, classes, q)
    from_train = class_quantiles(scores_train, y_train, classes, q)
    taus = {}
    for label in classes:
        key = normalize_label(label)
        n_val = int(np.sum(y_val_n == key))
        if n_val >= min_rows and key in from_val:
            taus[key] = from_val[key]
        elif key in from_train:
            taus[key] = from_train[key]
    known_val = scores_val[y_val_n != "UUC"]
    known_val = known_val[np.isfinite(known_val)]
    if known_val.size:
        fallback = float(linear_quantile(known_val, q))
    else:
        train = np.asarray(scores_train, dtype=float)
        train = train[np.isfinite(train)]
        fallback = float(linear_quantile(train, q)) if train.size else 0.0
    return taus, fallback


def apply_class_cuts(
    scores: np.ndarray,
    pred: Sequence[object],
    taus: dict[str, float],
    fallback: float,
) -> np.ndarray:
    """Reject when the score exceeds the cut of the predicted class."""
    labels = np.array([normalize_label(v) for v in np.asarray(pred).ravel()], dtype=object)
    tau = np.array([float(taus.get(label, fallback)) for label in labels], dtype=float)
    rejected = np.asarray(scores, dtype=float) > tau
    out = labels.copy()
    out[rejected] = "UUC"
    return out


def choose_class_quantile(
    scores_val: np.ndarray,
    pred_val: Sequence[object],
    y_val: Sequence[object],
    scores_train: np.ndarray,
    y_train: Sequence[object],
    classes: Sequence[object],
    known: Sequence[object],
) -> tuple[float, dict[str, float], float]:
    """Quantile with the higher paper micro-F1 on validation. Ties use ``select_q``."""
    scored: dict[float, float] = {}
    tables: dict[float, tuple[dict[str, float], float]] = {}
    for q in Q_GRID:
        taus, fallback = class_cut_table(scores_val, y_val, scores_train, y_train, classes, float(q))
        guessed = apply_class_cuts(scores_val, pred_val, taus, fallback)
        scored[float(q)] = float(micro_f1_reference(y_val, guessed, known))
        tables[float(q)] = (taus, fallback)
    chosen = float(select_q(scored))
    taus, fallback = tables[chosen]
    return chosen, taus, fallback


def combine_rejects(pred: Sequence[object], gng_reject: np.ndarray, svm_reject: np.ndarray, mode: str) -> np.ndarray:
    """``either`` rejects on one positive gate. ``both`` rejects only when both fire."""
    labels = np.array([normalize_label(v) for v in np.asarray(pred).ravel()], dtype=object)
    gng_reject = np.asarray(gng_reject, dtype=bool)
    svm_reject = np.asarray(svm_reject, dtype=bool)
    if mode == "either":
        reject = gng_reject | svm_reject
    elif mode == "both":
        reject = gng_reject & svm_reject
    else:
        raise ValueError(mode)
    labels = labels.copy()
    labels[reject] = "UUC"
    return labels


def _f1_at(scores: np.ndarray, pred: np.ndarray, y_true, known, tau: float) -> float:
    guessed = apply_threshold(scores, pred, float(tau))
    return float(micro_f1_reference(y_true, guessed, known))


def _svm_fit_predict(x_train, y_train, x_query, C: float, gamma, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train).astype(str)
    model = SVC(kernel="rbf", C=float(C), gamma=gamma, probability=True, random_state=int(seed))
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_query)
    score = 1.0 - proba.max(axis=1)
    pred = np.array([normalize_label(v) for v in model.classes_[np.argmax(proba, axis=1)]], dtype=object)
    return score, pred


def _landmark_fit_predict(x_train, y_train, x_query) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(np.asarray(x_train, dtype=float))
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(scaler.transform(x_train), np.asarray(y_train).astype(str))
    transformed = scaler.transform(x_query)
    proba = model.predict_proba(transformed)
    pred = np.array([normalize_label(v) for v in model.classes_[np.argmax(proba, axis=1)]], dtype=object)
    return 1.0 - proba.max(axis=1), pred


def _decision_metrics(y_true, guessed, pred_closed, known) -> dict[str, Any]:
    payload = _metrics(y_true, guessed, pred_closed, known)
    payload["tau"] = None
    payload["auroc"] = None
    return payload


def analyze_record(record: dict, json_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    """One saved network. The SVM hyperparameters come from the baseline JSON."""
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
    seed = int(record["seed"])
    svm_saved = _baseline_params(dataset, int(record["partition_id"]), nominal, "svm_prob")
    if svm_saved is None:
        raise RuntimeError(
            f"No svm_prob baseline for {dataset} partition {record['partition_id']} openness {nominal:.2f}."
        )

    nodes_s = _scale(nodes, mean, scale)
    x_fit = _scale(arrays["X_fit"], mean, scale)
    x_tr = _scale(arrays["X_tr"], mean, scale)
    x_te = _scale(arrays["X_te"], mean, scale)
    x_val = _scale(arrays["X_val_open"], mean, scale)
    y_fit = np.array([normalize_label(v) for v in arrays["y_fit"]], dtype=object)
    y_tr = np.array([normalize_label(v) for v in arrays["y_tr"]], dtype=object)
    y_tr_num = to_numeric(dataset, arrays["y_tr"])
    y_val = np.array([normalize_label(v) for v in arrays["y_val_open"]], dtype=object)
    node_num = relabel_from_nearest(nodes_s, x_tr, y_tr_num)
    known_num = to_numeric(dataset, known)
    vault = None if openness <= 0 else LabelVault(y_true, known)

    dist5_te, _, pred5_num = knn_predict(nodes_s, node_num, x_te, N_NEIGHBORS)
    dist5_tr, _, _ = knn_predict(nodes_s, node_num, x_tr, N_NEIGHBORS)
    dist5_val, _, pred5_val_num = knn_predict(nodes_s, node_num, x_val, N_NEIGHBORS)
    pred5 = closed_labels(dataset, pred5_num)
    pred5_val = closed_labels(dataset, pred5_val_num)

    rules: dict[str, Any] = {}
    if openness <= 0:
        cut_choice = "closed"
        cut_val_f1: dict[str, float] = {}
        rules["pick_cut"] = _pack_rule("pick_cut", dist5_te, pred5, y_true, known, openness, None, vault)
        for name in CUT_ORDER:
            rules[f"dist5__{name}"] = dict(rules["pick_cut"])
    else:
        tau_val = _select_val(dist5_val, pred5_val, y_val, fitting, openness)
        tau_q95 = float(linear_quantile(dist5_tr, COVERAGE))
        tau_copied = float(record["threshold_dist"])
        taus = {"val": float(tau_val), "train_q95": tau_q95, "copied": tau_copied}
        cut_val_f1 = {name: _f1_at(dist5_val, pred5_val, y_val, fitting, taus[name]) for name in CUT_ORDER}
        cut_choice = choose_cut(cut_val_f1, taus, dist5_val)
        for name in CUT_ORDER:
            rules[f"dist5__{name}"] = _pack_rule("dist5", dist5_te, pred5, y_true, known, openness, taus[name], vault)
        rules["pick_cut"] = dict(rules[f"dist5__{cut_choice}"])
        rules["pick_cut"]["chosen"] = cut_choice

    c_value = svm_saved["selected_params"]["C"]
    gamma = svm_saved["selected_params"]["gamma"]
    svm_val, svm_pred_val = _svm_fit_predict(x_fit, y_fit, x_val, c_value, gamma, seed)
    svm_te, svm_pred_te = _svm_fit_predict(x_tr, y_tr, x_te, c_value, gamma, seed)
    gng_tau = _select_val(dist5_val, svm_pred_val, y_val, fitting, openness)
    svm_tau = _select_val(svm_val, svm_pred_val, y_val, fitting, openness)
    rules["svm_label__gng_val"] = _pack_rule(
        "svm_label__gng_val", dist5_te, svm_pred_te, y_true, known, openness, gng_tau, vault
    )
    rules["svm_label__svm_val"] = _pack_rule(
        "svm_label__svm_val", svm_te, svm_pred_te, y_true, known, openness, svm_tau, vault
    )

    if openness <= 0:
        gng_reject = np.zeros(dist5_te.shape[0], dtype=bool)
        svm_reject = np.zeros(svm_te.shape[0], dtype=bool)
    else:
        gng_reject = dist5_te > float(gng_tau)
        svm_reject = svm_te > float(svm_tau)
    for mode, rule_name in (("either", "either_rejects"), ("both", "both_rejects")):
        guessed = combine_rejects(svm_pred_te, gng_reject, svm_reject, mode)
        rules[rule_name] = _decision_metrics(y_true, guessed, svm_pred_te, known)

    landmarks = {
        "fit": finite_landmarks(class_landmarks(nodes_s, node_num, x_fit, known_num)),
        "train": finite_landmarks(class_landmarks(nodes_s, node_num, x_tr, known_num)),
        "val": finite_landmarks(class_landmarks(nodes_s, node_num, x_val, known_num)),
        "test": finite_landmarks(class_landmarks(nodes_s, node_num, x_te, known_num)),
    }
    proba_val, land_pred_val = _landmark_fit_predict(landmarks["fit"], y_fit, landmarks["val"])
    proba_te, land_pred_te = _landmark_fit_predict(landmarks["train"], y_tr, landmarks["test"])
    proba_tr, _ = _landmark_fit_predict(landmarks["train"], y_tr, landmarks["train"])
    min_val = landmarks["val"].min(axis=1)
    min_te = landmarks["test"].min(axis=1)
    min_tr = landmarks["train"].min(axis=1)
    for stem, score_te, score_tr, score_val, pred_te, pred_val in (
        ("landmark__proba", proba_te, proba_tr, proba_val, land_pred_te, land_pred_val),
        ("landmark__min", min_te, min_tr, min_val, land_pred_te, land_pred_val),
    ):
        tau_val = _select_val(score_val, pred_val, y_val, fitting, openness)
        tau_tr = None if openness <= 0 else float(linear_quantile(score_tr, COVERAGE))
        rules[f"{stem}__val"] = _pack_rule(stem, score_te, pred_te, y_true, known, openness, tau_val, vault)
        rules[f"{stem}__train_q95"] = _pack_rule(stem, score_te, pred_te, y_true, known, openness, tau_tr, vault)

    per_class_q = None
    if openness <= 0:
        rules["per_class__dist5"] = _pack_rule("per_class__dist5", dist5_te, pred5, y_true, known, openness, None, vault)
    else:
        per_class_q, taus_c, fallback = choose_class_quantile(
            dist5_val, pred5_val, y_val, dist5_tr, y_tr, known, fitting
        )
        guessed = apply_class_cuts(dist5_te, pred5, taus_c, fallback)
        rules["per_class__dist5"] = _decision_metrics(y_true, guessed, pred5, known)
        rules["per_class__dist5"]["q"] = float(per_class_q)
        rules["per_class__dist5"]["auroc"] = rules["dist5__val"]["auroc"]

    for k in KS:
        mean_te, kth_te, pred_num = knn_distance_weighted(nodes_s, node_num, x_te, k)
        mean_tr, kth_tr, _ = knn_distance_weighted(nodes_s, node_num, x_tr, k)
        mean_val, kth_val, pred_val_num = knn_distance_weighted(nodes_s, node_num, x_val, k)
        pred = closed_labels(dataset, pred_num)
        pred_val = closed_labels(dataset, pred_val_num)
        for stem, score_te, score_tr, score_val in (
            (f"knn_w{k}__mean", mean_te, mean_tr, mean_val),
            (f"knn_w{k}__kth", kth_te, kth_tr, kth_val),
        ):
            tau_val = _select_val(score_val, pred_val, y_val, fitting, openness)
            tau_tr = None if openness <= 0 else float(linear_quantile(score_tr, COVERAGE))
            rules[f"{stem}__val"] = _pack_rule(stem, score_te, pred, y_true, known, openness, tau_val, vault)
            rules[f"{stem}__train_q95"] = _pack_rule(stem, score_te, pred, y_true, known, openness, tau_tr, vault)

    return {
        "ok": True,
        "dataset": dataset,
        "partition_id": int(record["partition_id"]),
        "nominal_openness": nominal,
        "actual_openness_percent": openness,
        "n_nodes": int(nodes_s.shape[0]),
        "cut_choice": cut_choice,
        "cut_val_f1": {name: float(value) for name, value in cut_val_f1.items()},
        "per_class_q": None if per_class_q is None else float(per_class_q),
        "svm_C": float(c_value),
        "svm_gamma": gamma if isinstance(gamma, str) else float(gamma),
        "rules": rules,
    }


HEADLINE = [
    "pick_cut",
    "svm_label__gng_val",
    "landmark__proba__val",
    "landmark__min__val",
    "per_class__dist5",
    "either_rejects",
    "both_rejects",
]
CUTS = ["dist5__val", "dist5__train_q95", "dist5__copied"]
NEIGHBORS = [f"knn_w{k}__{score}__val" for k in KS for score in ("mean", "kth")]
EXTRA = (
    ["svm_label__svm_val", "landmark__proba__train_q95", "landmark__min__train_q95"]
    + [f"knn_w{k}__{score}__train_q95" for k in KS for score in ("mean", "kth")]
)
ALL_RULES = HEADLINE + CUTS + NEIGHBORS + EXTRA


def write_report(rows: list[dict]) -> None:
    keys = sorted({(r["dataset"], round(float(r["nominal_openness"]), 2)) for r in rows})
    lines = [
        "# GNG options",
        "",
        "No gas is grown. Nodes are the saved Phase 1 network, relabeled from the",
        "nearest training row. Paper micro-F1 is the mean across partitions.",
        "",
        "`pick` deploys the 5-NN distance cut with the higher validation F1 among",
        "the validation sweep, the training-row 0.95 quantile, and the copied",
        "Phase 1 distance threshold. `choice` counts which cut won.",
        "`SVM+GNG` uses the selected raw-feature SVM for the label and the 5-NN",
        "distance, cut on validation, for rejection.",
        "`landmark proba` is a logistic regression on the distance to the nearest",
        "node of each class. `landmark min` rejects on the smallest of those distances.",
        "`per class` uses one distance quantile per class. The quantile is chosen",
        "on validation.",
        "`either` rejects when the GNG gate or the SVM probability gate rejects.",
        "`both` rejects only when both gates reject. The label is the SVM label.",
        "",
        "| Dataset | O | n | choice | pick | SVM+GNG | landmark proba | landmark min | per class | either | both |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cells = []
    for dataset, nominal in keys:
        group = [r for r in rows if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        counts = Counter(str(r.get("cut_choice", "")) for r in group)
        choice = " ".join(f"{name}:{counts[name]}" for name in sorted(counts))
        cell: dict[str, Any] = {
            "dataset": dataset,
            "nominal_openness": nominal,
            "n": len(group),
            "choice": choice,
        }
        for rule in ALL_RULES:
            for metric, digits in (("paper_f1_percent", 2), ("FRR", 3), ("FAR", 3), ("closed_acc", 3), ("auroc", 3)):
                mean, std = _mean_std(r["rules"].get(rule, {}).get(metric) for r in group)
                cell[f"{rule}_{metric}"] = _fmt(mean, std, digits)
        cell["per_class_q"] = _fmt(*_mean_std(r.get("per_class_q") for r in group), 3)
        cells.append(cell)
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {n} | {choice} | {pick_cut_paper_f1_percent} | "
            "{svm_label__gng_val_paper_f1_percent} | {landmark__proba__val_paper_f1_percent} | "
            "{landmark__min__val_paper_f1_percent} | {per_class__dist5_paper_f1_percent} | "
            "{either_rejects_paper_f1_percent} | {both_rejects_paper_f1_percent} |".format(**cell)
        )
    lines.extend(
        [
            "",
            "| Dataset | O | val cut | train q95 | copied distance | pick FRR | pick FAR | SVM+GNG FRR | SVM+GNG FAR | either FRR | either FAR | both FRR | both FAR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {dist5__val_paper_f1_percent} | "
            "{dist5__train_q95_paper_f1_percent} | {dist5__copied_paper_f1_percent} | "
            "{pick_cut_FRR} | {pick_cut_FAR} | {svm_label__gng_val_FRR} | {svm_label__gng_val_FAR} | "
            "{either_rejects_FRR} | {either_rejects_FAR} | {both_rejects_FRR} | {both_rejects_FAR} |".format(**cell)
        )
    lines.extend(
        [
            "",
            "Distance-weighted k-NN. The cut maximises validation paper micro-F1.",
            "`mean` is the mean distance to the k neighbors. `kth` is the distance to the k-th neighbor.",
            "",
            "| Dataset | O | k3 mean | k3 kth | k7 mean | k7 kth | k11 mean | k11 kth |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {knn_w3__mean__val_paper_f1_percent} | "
            "{knn_w3__kth__val_paper_f1_percent} | {knn_w7__mean__val_paper_f1_percent} | "
            "{knn_w7__kth__val_paper_f1_percent} | {knn_w11__mean__val_paper_f1_percent} | "
            "{knn_w11__kth__val_paper_f1_percent} |".format(**cell)
        )
    lines.append("")
    path = RESULTS_STATIC / "GNG_OPTIONS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    table = RESULTS_STATIC / "tables" / "gng_options.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "nominal_openness", "n", "choice", "per_class_q"]
    for rule in ALL_RULES:
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
        / "gng_options"
        / "raw"
        / record["dataset"]
        / f"p{record['partition_id']}"
        / f"O{nominal}.json"
    )
    if dest.is_file():
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if existing.get("ok") and "both_rejects" in existing.get("rules", {}) and "knn_w11__kth__val" in existing.get("rules", {}):
            return existing
    bundle = _load_bundle(record["dataset"])
    payload = analyze_record(record, path, bundle)
    atomic_json(dest, payload)
    return payload


def run_gng_options(only: dict[str, str] | None = None, jobs: int = 1, resume: bool = True) -> dict[str, Any]:
    """Score every saved Phase 1 network and write ``results/GNG_OPTIONS.md``."""
    del resume
    records = _phase1_records(RESULTS_STATIC / "phase1")
    selected = [path for record, path in records if _matches(record, only)]
    if not selected:
        raise SystemExit("no Phase 1 models matched")
    rows = []
    workers = max(1, int(jobs))
    print(f"{len(selected)} GNG-option jobs, {workers} workers", flush=True)
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
    return {"n": len(rows), "report": str(RESULTS_STATIC / "GNG_OPTIONS.md")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Six rescores of saved GNG nodes. No gas is trained.")
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
    print(json.dumps(run_gng_options(only or None, jobs=args.jobs, resume=args.resume)), flush=True)


if __name__ == "__main__":
    main()
