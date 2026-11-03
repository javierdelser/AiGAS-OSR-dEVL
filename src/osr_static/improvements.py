"""Rescore the saved GNG networks. No Growing Neural Gas is trained.

Three checks from the gap assessment:

1. Relabel each saved node with its nearest training row, then classify with
   5-NN and with 1-NN. The legacy node vote uses k equal to the class count.
2. Replace the absolute 5-NN distance with an OSNN ratio on the nodes, and
   with distance divided by the local node spacing. The cut is chosen on the
   open validation split using only nodes of the fitting classes, then applied
   to the full deployed network.
3. Keep the saved 1_NN reject decision and take the known-class label from
   the distance k-NN or the RBF SVM already selected on that partition.

The training-row 95% distance cut is reported again on the relabeled network.
It does not use test labels. It is in-sample, because those rows grew the network.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from osr_static.baselines.knn_dist import scores as knn_scores
from osr_static.baselines.osnn import scores as osnn_scores
from osr_static.baselines.svm_prob import fit_predict as svm_fit_predict
from osr_static.data import load_txt_any
from osr_static.jobs import atomic_json, dataset_file, partition_file
from osr_static.paths import RESULTS_STATIC
from osr_static.metrics import auroc_unknown, decompose, micro_f1_reference, normalize_label
from osr_static.partitions import find_record, load_partitions, materialize
from osr_static.scores import linear_quantile
from osr_static.selection import select_threshold
from osr_static.threshold_transfer import COVERAGE, apply_gates

DATA_FILES = {
    "LETTER": "letter_dataset.csv",
    "PENDIGITS": "pendigits_dataset.csv",
}
N_NEIGHBORS = 5


def to_numeric(dataset: str, labels: Sequence[object]) -> np.ndarray:
    """Letter A–Z → 1–26, Pendigits digit strings → ints, UUC → 100."""
    out = []
    for value in np.asarray(labels).ravel():
        text = normalize_label(value)
        if text == "UUC":
            out.append(100)
        elif dataset == "LETTER":
            out.append(ord(text) - ord("A") + 1)
        else:
            out.append(int(float(text)))
    return np.asarray(out, dtype=int)


def closed_labels(dataset: str, pred: np.ndarray) -> np.ndarray:
    labels = []
    for value in np.asarray(pred).ravel():
        number = int(value)
        if number == 100:
            labels.append("UUC")
        elif dataset == "LETTER":
            labels.append(chr(number - 1 + ord("A")))
        else:
            labels.append(str(number))
    return np.asarray(labels, dtype=object)


def relabel_from_nearest(nodes: np.ndarray, rows: np.ndarray, row_labels: np.ndarray) -> np.ndarray:
    """Label of the nearest row, one neighbour, in the space of ``nodes`` and ``rows``."""
    model = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    model.fit(np.asarray(rows, dtype=float))
    _, index = model.kneighbors(np.asarray(nodes, dtype=float))
    return np.asarray(row_labels)[index.ravel()]


def local_spacing(nodes: np.ndarray, k: int = N_NEIGHBORS) -> np.ndarray:
    """Mean distance from each node to its ``k`` other nodes. Self-distance is dropped."""
    nodes = np.asarray(nodes, dtype=float)
    n_neighbors = min(k + 1, nodes.shape[0])
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    model.fit(nodes)
    distances, _ = model.kneighbors(nodes)
    if distances.shape[1] == 1:
        return np.ones(nodes.shape[0], dtype=float)
    return distances[:, 1:].mean(axis=1)


def spacing_scores(
    nodes: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    spacing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``distance to nearest node / that node's spacing``. Higher is more unknown."""
    model = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    model.fit(np.asarray(nodes, dtype=float))
    distances, index = model.kneighbors(np.asarray(queries, dtype=float))
    index = index.ravel()
    score = distances.ravel() / np.maximum(np.asarray(spacing, dtype=float)[index], 1e-12)
    return score, np.asarray(labels)[index]


def knn_predict(
    nodes: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean distance to ``k`` neighbours, max probability, and the integer class."""
    nodes = np.asarray(nodes, dtype=float)
    queries = np.asarray(queries, dtype=float)
    k_use = min(int(k), nodes.shape[0])
    model = KNeighborsClassifier(n_neighbors=k_use, algorithm="ball_tree")
    model.fit(nodes, np.asarray(labels).ravel())
    distances, _ = model.kneighbors(queries)
    pred = np.rint(model.predict(queries).astype(float)).astype(int)
    confidence = model.predict_proba(queries).max(axis=1)
    return distances.mean(axis=1), confidence, pred


def _scale(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=float).copy()
    scale[scale == 0] = 1.0
    return (np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)) / scale


def _metrics(y_true, pred, pred_closed, known) -> dict[str, float | None]:
    parts = decompose(y_true, pred, pred_closed, known)
    return {
        "paper_f1_percent": float(100.0 * micro_f1_reference(y_true, pred, known)),
        "FRR": float(parts["FRR"]),
        "FAR": None if parts["n_unknown"] == 0 else float(parts["FAR"]),
        "closed_acc": float(parts["closed_acc"]),
    }


def _phase1_records(phase1: Path) -> list[tuple[dict, Path]]:
    found = []
    for path in sorted(phase1.glob("raw/phase1/*/*/*/1_NN__*.json")):
        if path.name.endswith(".job.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("ok"):
            found.append((record, path))
    return found


def _baseline_params(dataset: str, partition_id: int, nominal: float, method: str) -> dict | None:
    nominal_name = f"{nominal:.2f}"
    path = (
        RESULTS_STATIC
        / "baselines"
        / "raw"
        / dataset
        / f"p{partition_id}"
        / f"O{nominal_name}"
        / f"{method}.json"
    )
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        return None
    return payload


def _select_open(scores, pred, y_val, known_fit, openness: float) -> float | None:
    if openness <= 0:
        return None
    tau, _ = select_threshold(scores, pred, y_val, known_fit)
    return float(tau)


def _select_osnn_grid(scores, pred, y_val, known_fit, openness: float) -> float | None:
    """Same 51 cuts in ``[0.5, 1]`` as the OSNN baseline. Ties prefer 0.75."""
    if openness <= 0:
        return None
    from osr_static.metrics import apply_threshold

    best = (-1.0, 0.75)
    for threshold in np.linspace(0.5, 1.0, 51):
        guessed = apply_threshold(scores, pred, float(threshold))
        f1 = micro_f1_reference(y_val, guessed, known_fit)
        closer = abs(float(threshold) - 0.75) < abs(best[1] - 0.75)
        if f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and closer):
            best = (float(f1), float(threshold))
    return best[1]


def analyze_record(record: dict, json_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    """One saved network. Raises if the legacy copied cut does not reproduce its paper F1."""
    dataset = record["dataset"]
    arrays = materialize(
        bundle["x"],
        bundle["y"],
        find_record(bundle["document"], record["partition_id"], record["nominal_openness"]),
    )
    with np.load(json_path.with_suffix(".npz"), allow_pickle=True) as saved:
        saved_dist = np.asarray(saved["dist_mean"], dtype=float)
        saved_conf = np.asarray(saved["confidence"], dtype=float)
        y_true = np.array([normalize_label(v) for v in saved["y_true"]], dtype=object)
        pred_legacy = closed_labels(dataset, saved["pred_base"])
        nodes = np.asarray(saved["node_positions"], dtype=float)
        node_labels = np.rint(np.asarray(saved["node_labels"], dtype=float)).astype(int)
        mean = np.asarray(saved["scaler_mean"], dtype=float)
        scale = np.asarray(saved["scaler_scale"], dtype=float)

    openness = float(record["actual_openness_percent"])
    dist_thr = float(record["threshold_dist"])
    conf_thr = float(record["threshold_conf"])
    known = [str(v) for v in record["known_classes"]]
    fitting = [str(v) for v in record["fitting_classes"]]
    fitting_num = set(int(v) for v in to_numeric(dataset, fitting))

    nodes_s = _scale(nodes, mean, scale)
    x_tr = _scale(arrays["X_tr"], mean, scale)
    x_te = _scale(arrays["X_te"], mean, scale)
    x_val = _scale(arrays["X_val_open"], mean, scale)
    y_tr_num = to_numeric(dataset, arrays["y_tr"])
    y_tr = np.array([normalize_label(v) for v in arrays["y_tr"]], dtype=object)
    y_val = np.array([normalize_label(v) for v in arrays["y_val_open"]], dtype=object)

    legacy_pred = apply_gates(saved_dist, saved_conf, pred_legacy, dist_thr, conf_thr, openness)
    legacy = _metrics(y_true, legacy_pred, pred_legacy, known)
    if abs(legacy["paper_f1_percent"] - float(record["paper_f1_percent"])) > 1e-4:
        raise RuntimeError(f"copied cut does not reproduce the stored paper F1 for {json_path}")

    new_num = relabel_from_nearest(nodes_s, x_tr, y_tr_num)
    agree = float(np.mean(new_num == node_labels))

    dist5, conf5, pred5 = knn_predict(nodes_s, new_num, x_te, N_NEIGHBORS)
    dist_gap = float(np.max(np.abs(dist5 - saved_dist))) if saved_dist.size else 0.0
    if dist_gap > 1e-4:
        raise RuntimeError(f"relabeled 5-NN distances differ from the saved scores by {dist_gap} for {json_path}")
    pred5_s = closed_labels(dataset, pred5)
    dist_tr, _, _ = knn_predict(nodes_s, new_num, x_tr, N_NEIGHBORS)
    train_q95 = float(linear_quantile(dist_tr, COVERAGE))
    _, _, pred1 = knn_predict(nodes_s, new_num, x_te, 1)

    rules: dict[str, dict] = {
        "legacy_copied": legacy,
        "relabel_k5_closed": _metrics(y_true, pred5_s, pred5_s, known),
        "relabel_k1_closed": _metrics(y_true, closed_labels(dataset, pred1), closed_labels(dataset, pred1), known),
        "relabel_k5_copied": _metrics(
            y_true,
            apply_gates(dist5, conf5, pred5_s, dist_thr, conf_thr, openness),
            pred5_s,
            known,
        ),
        "relabel_k5_train_q95": _metrics(
            y_true,
            apply_gates(dist5, conf5, pred5_s, train_q95, 0.0, openness),
            pred5_s,
            known,
        ),
    }

    def _ratio_rule(labels_num: np.ndarray, name: str, selector) -> None:
        labels_s = closed_labels(dataset, labels_num)
        fit_mask = np.isin(labels_num, list(fitting_num))
        if int(fit_mask.sum()) < 2:
            rules[name] = {"paper_f1_percent": None, "FRR": None, "FAR": None, "closed_acc": None, "tau": None}
            return
        score_val, pred_val = osnn_scores(nodes_s[fit_mask], labels_s[fit_mask], x_val)
        tau = selector(score_val, pred_val, y_val, fitting, openness)
        score_te, pred_te = osnn_scores(nodes_s, labels_s, x_te)
        if tau is None:
            guessed = pred_te
        else:
            guessed = apply_gates_score(score_te, pred_te, tau)
        metrics = _metrics(y_true, guessed, pred_te, known)
        metrics["tau"] = tau
        unknown = ~np.isin(y_true, [normalize_label(v) for v in known])
        metrics["auroc"] = None if openness <= 0 else auroc_unknown(score_te, unknown)
        rules[name] = metrics

    def _spacing_rule(labels_num: np.ndarray, name: str) -> None:
        labels_s = closed_labels(dataset, labels_num)
        spacing = local_spacing(nodes_s)
        fit_mask = np.isin(labels_num, list(fitting_num))
        score_val, pred_val = spacing_scores(nodes_s[fit_mask], labels_s[fit_mask], x_val, spacing[fit_mask])
        tau = _select_open(score_val, pred_val, y_val, fitting, openness)
        score_te, pred_te = spacing_scores(nodes_s, labels_s, x_te, spacing)
        guessed = pred_te if tau is None else apply_gates_score(score_te, pred_te, tau)
        metrics = _metrics(y_true, guessed, pred_te, known)
        metrics["tau"] = tau
        unknown = ~np.isin(y_true, [normalize_label(v) for v in known])
        metrics["auroc"] = None if openness <= 0 else auroc_unknown(score_te, unknown)
        rules[name] = metrics

    _ratio_rule(node_labels, "ratio_legacy_fit", _select_osnn_grid)
    _ratio_rule(new_num, "ratio_relabel_fit", _select_osnn_grid)
    _ratio_rule(new_num, "ratio_relabel_free", _select_open)
    _spacing_rule(new_num, "spacing_relabel_fit")

    nominal = float(record["nominal_openness"])
    partition_id = int(record["partition_id"])
    knn_meta = _baseline_params(dataset, partition_id, nominal, "knn_dist")
    svm_meta = _baseline_params(dataset, partition_id, nominal, "svm_prob")
    hybrid_notes = {}
    if knn_meta is not None:
        k = int(knn_meta["selected_params"]["k"])
        _, knn_pred = knn_scores(x_tr, y_tr, x_te, k)
        knn_pred = np.array([normalize_label(v) for v in knn_pred], dtype=object)
        rules["hybrid_knn_copied"] = _metrics(
            y_true, apply_gates(saved_dist, saved_conf, knn_pred, dist_thr, conf_thr, openness), knn_pred, known
        )
        rules["hybrid_knn_train_q95"] = _metrics(
            y_true, apply_gates(saved_dist, saved_conf, knn_pred, train_q95, 0.0, openness), knn_pred, known
        )
        hybrid_notes["knn_k"] = k
        hybrid_notes["knn_closed_acc"] = rules["hybrid_knn_copied"]["closed_acc"]
        hybrid_notes["knn_baseline_closed_acc"] = float(knn_meta["decomposition"]["closed_acc"])
    if svm_meta is not None:
        params = svm_meta["selected_params"]
        _, svm_pred = svm_fit_predict(x_tr, y_tr, x_te, params["C"], params["gamma"])
        svm_pred = np.array([normalize_label(v) for v in svm_pred], dtype=object)
        rules["hybrid_svm_copied"] = _metrics(
            y_true, apply_gates(saved_dist, saved_conf, svm_pred, dist_thr, conf_thr, openness), svm_pred, known
        )
        rules["hybrid_svm_train_q95"] = _metrics(
            y_true, apply_gates(saved_dist, saved_conf, svm_pred, train_q95, 0.0, openness), svm_pred, known
        )
        hybrid_notes["svm_C"] = params["C"]
        hybrid_notes["svm_gamma"] = params["gamma"]
        hybrid_notes["svm_closed_acc"] = rules["hybrid_svm_copied"]["closed_acc"]
        hybrid_notes["svm_baseline_closed_acc"] = float(svm_meta["decomposition"]["closed_acc"])

    return {
        "ok": True,
        "dataset": dataset,
        "partition_id": partition_id,
        "nominal_openness": nominal,
        "actual_openness_percent": openness,
        "n_nodes": int(nodes.shape[0]),
        "label_agreement": agree,
        "train_q95": train_q95,
        "rules": rules,
        "hybrid": hybrid_notes,
    }


def apply_gates_score(scores: np.ndarray, pred: np.ndarray, tau: float) -> np.ndarray:
    """Accept when ``score <= tau``. Higher scores are more unknown."""
    from osr_static.metrics import apply_threshold

    return apply_threshold(scores, pred, tau)


def _mean_std(values) -> tuple[float | None, float | None]:
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            clean.append(number)
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=float)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def _fmt(mean: float | None, std: float | None, digits: int = 2) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_report(rows: list[dict]) -> None:
    """Cell means for the rescore, next to the other result tables."""
    keys = sorted({(r["dataset"], round(float(r["nominal_openness"]), 2)) for r in rows})
    rule_names = [
        "legacy_copied",
        "relabel_k5_closed",
        "relabel_k1_closed",
        "relabel_k5_copied",
        "relabel_k5_train_q95",
        "ratio_legacy_fit",
        "ratio_relabel_fit",
        "ratio_relabel_free",
        "spacing_relabel_fit",
        "hybrid_knn_copied",
        "hybrid_svm_copied",
        "hybrid_knn_train_q95",
        "hybrid_svm_train_q95",
    ]
    cells = []
    for dataset, nominal in keys:
        group = [r for r in rows if r["dataset"] == dataset and abs(float(r["nominal_openness"]) - nominal) < 1e-6]
        cell: dict[str, Any] = {
            "dataset": dataset,
            "nominal_openness": nominal,
            "n": len(group),
            "label_agreement": _fmt(*_mean_std(r["label_agreement"] for r in group), 3),
        }
        for rule in rule_names:
            for metric, digits in (("paper_f1_percent", 2), ("FRR", 3), ("FAR", 3), ("closed_acc", 3)):
                mean, std = _mean_std(r["rules"].get(rule, {}).get(metric) for r in group)
                cell[f"{rule}_{metric}"] = _fmt(mean, std, digits)
        cells.append(cell)

    lines = [
        "# Saved-network improvements",
        "",
        "No Growing Neural Gas was retrained. `legacy_copied` is the Phase 1 operating point.",
        "Relabel assigns each node the class of its nearest training row.",
        "`relabel_k5_closed` and `relabel_k1_closed` leave rejection off, so on open cells",
        "their paper micro-F1 counts every accepted unknown. Read `closed_acc` there.",
        "The ratio cut is chosen from the same 51 values in [0.5, 1] as OSNN, on the open",
        "validation split, using only nodes whose labels are fitting classes, then applied",
        "to every node. `ratio_relabel_free` in the CSV is the unrestricted sweep of that ratio.",
        "The spacing score uses the unrestricted sweep.",
        "Hybrid rows keep the saved 1_NN reject mask and replace the known-class label.",
        "`train_q95` is the 0.95 quantile of training-row distances on this network.",
        "",
        "| Dataset | O | Agreement | Legacy F1 | Relabel k5, no reject, closed acc | Relabel k1, no reject, closed acc | Relabel k5 + copied F1 | Relabel k5 + train q95 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {label_agreement} | {legacy_copied_paper_f1_percent} | "
            "{relabel_k5_closed_closed_acc} | {relabel_k1_closed_closed_acc} | "
            "{relabel_k5_copied_paper_f1_percent} | {relabel_k5_train_q95_paper_f1_percent} |".format(**cell)
        )
    lines.extend(
        [
            "",
            "| Dataset | O | Ratio, legacy labels | Ratio, relabel | Spacing, relabel | k-NN label + copied reject | SVM label + copied reject | SVM label + train q95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {ratio_legacy_fit_paper_f1_percent} | "
            "{ratio_relabel_fit_paper_f1_percent} | {spacing_relabel_fit_paper_f1_percent} | "
            "{hybrid_knn_copied_paper_f1_percent} | {hybrid_svm_copied_paper_f1_percent} | "
            "{hybrid_svm_train_q95_paper_f1_percent} |".format(**cell)
        )
    lines.extend(
        [
            "",
            "| Dataset | O | Legacy FAR | Relabel + train q95 FAR | Ratio relabel FAR | Hybrid SVM copied FAR | Hybrid SVM copied FRR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {dataset} | {nominal_openness:.2f} | {legacy_copied_FAR} | {relabel_k5_train_q95_FAR} | "
            "{ratio_relabel_fit_FAR} | {hybrid_svm_copied_FAR} | {hybrid_svm_copied_FRR} |".format(**cell)
        )
    lines.append("")
    path = RESULTS_STATIC / "IMPROVEMENTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    table = RESULTS_STATIC / "tables" / "improvements.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "nominal_openness", "n", "label_agreement"]
    for rule in rule_names:
        for metric in ("paper_f1_percent", "FRR", "FAR", "closed_acc"):
            columns.append(f"{rule}_{metric}")
    with table.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for cell in cells:
            handle.write(",".join(str(cell.get(col, "")) for col in columns) + "\n")


def _matches(record: dict, only: dict[str, str] | None) -> bool:
    if not only:
        return True
    if "dataset" in only and record["dataset"].lower() != only["dataset"].lower():
        return False
    if "openness" in only and abs(float(record["nominal_openness"]) - float(only["openness"])) > 1e-6:
        return False
    if "partition" in only and int(record["partition_id"]) != int(only["partition"]):
        return False
    return True


def _load_bundle(dataset: str) -> dict[str, Any]:
    x, y = load_txt_any(dataset_file(DATA_FILES[dataset]))
    return {"x": x, "y": y, "document": load_partitions(partition_file(dataset))}


def score_one(json_path: str) -> dict[str, Any]:
    """Worker entry. Loads the dataset itself so processes do not pickle the tables."""
    path = Path(json_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    nominal = f"{float(record['nominal_openness']):.2f}"
    dest = (
        RESULTS_STATIC
        / "improvements"
        / "raw"
        / record["dataset"]
        / f"p{record['partition_id']}"
        / f"O{nominal}.json"
    )
    if dest.is_file():
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if existing.get("ok") and "ratio_relabel_fit" in existing.get("rules", {}) and "tau" in existing["rules"]["ratio_relabel_fit"]:
            # Older files from the unrestricted sweep stored a near-zero tau under this name.
            tau = existing["rules"]["ratio_relabel_fit"].get("tau")
            if tau is None or float(tau) >= 0.5 or float(existing.get("actual_openness_percent", 1)) <= 0:
                return existing
    bundle = _load_bundle(record["dataset"])
    payload = analyze_record(record, path, bundle)
    atomic_json(dest, payload)
    return payload


def run_improvements(only: dict[str, str] | None = None, jobs: int = 1, resume: bool = True) -> dict[str, Any]:
    """Score every saved Phase 1 network and write ``results/IMPROVEMENTS.md``."""
    del resume
    records = _phase1_records(RESULTS_STATIC / "phase1")
    selected = [path for record, path in records if _matches(record, only)]
    if not selected:
        raise SystemExit("no Phase 1 models matched")
    rows = []
    workers = max(1, int(jobs))
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
    return {"n": len(rows), "report": str(RESULTS_STATIC / "IMPROVEMENTS.md")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rescore saved 1_NN networks. Does not train a GNG.")
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
    result = run_improvements(only or None, jobs=args.jobs, resume=args.resume)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
