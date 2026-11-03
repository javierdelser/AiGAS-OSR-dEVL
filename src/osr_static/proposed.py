"""Run the legacy GNG + 1_NN pipeline and keep the arrays the script does not return.

The training, node labeling, scaler, threshold search and final ``1_NN`` call are
``run_method`` itself. This module only seeds NumPy, calls that function, and
recomputes the 5-NN distances so the reject decision can be checked.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from osr_static.metrics import decompose, micro_f1_reference, micro_f1_repo
from osr_static.partitions import find_record, load_partitions, materialize
from osr_static.scores import reject_mask_1nn, score_1nn


def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    """Train one partition and write its JSON and ``.npz``. Returns the scalar record."""
    out_json = Path(job["out_json"])
    out_npz = Path(job["out_npz"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        record = _run(job)
    except Exception as exc:
        failed = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "job": {k: job[k] for k in job if k != "env"},
        }
        failed_path = Path(job["failed_json"])
        _atomic_json(failed_path, failed)
        raise
    record["wall_seconds"] = time.time() - started
    _atomic_json(out_json, record)
    return record


def _run(job: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(job["repo_root"])
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # osr_utils.parse_args() runs at import and rejects this worker's flags.
    sys.argv = [sys.argv[0]]

    import osr_GNG_Train_VAl_Test_LOOP_Split26 as loop
    from osr_static.data import load_txt_any

    document = load_partitions(Path(job["partition_file"]))
    part = find_record(document, int(job["partition_id"]), float(job["nominal_openness"]))
    data_path = repo_root / "DATA" / job["data_file"]
    x, y = load_txt_any(data_path)
    arrays = materialize(x, y, part)
    splits = loop.OSRSplits(
        arrays["X_fit"],
        arrays["y_fit"],
        arrays["X_val_closed"],
        arrays["y_val_closed"],
        arrays["X_val_open"],
        arrays["y_val_open"],
        arrays["X_te"],
        arrays["y_te"],
        arrays["kkc_labels_fit"],
        arrays["kkc_labels_all"],
        arrays["X_tr"],
        arrays["y_tr"],
    )
    flag = 0 if job["dataset"] == "LETTER" else 1
    actual_openness_percent = 100.0 * loop.osr.openness(part["c_tr"], part["c_te"])

    np.random.seed(int(job["seed"]))
    captured: dict[str, Any] = {}
    gng = loop.GNG_Funciones
    original = gng.funcion_predecir_scoresREAL

    def _capture(Xfeatures, Xlabels, test, metrica, openess, umbralDIST, umbralCONF, scaler=None, XTRAIN=None, YTRAIN=None):
        pred, score = original(
            Xfeatures, Xlabels, test, metrica, openess, umbralDIST, umbralCONF,
            scaler=scaler, XTRAIN=XTRAIN, YTRAIN=YTRAIN,
        )
        captured["Xfeatures"] = np.asarray(Xfeatures, dtype=float)
        captured["Xlabels"] = np.asarray(Xlabels).copy()
        captured["test"] = np.asarray(test, dtype=float)
        captured["openess"] = float(openess)
        captured["umbralDIST"] = float(umbralDIST)
        captured["umbralCONF"] = float(umbralCONF)
        captured["scaler_mean"] = np.asarray(scaler.mean_, dtype=float)
        captured["scaler_scale"] = np.asarray(scaler.scale_, dtype=float)
        captured["pred"] = np.asarray(pred).copy()
        captured["legacy_score"] = np.asarray(score, dtype=float).copy()
        return pred, score

    gng.funcion_predecir_scoresREAL = _capture
    auroc_prefix = str(Path(job["auroc_prefix"]))
    Path(auroc_prefix).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        repo_f1_percent, acc_percent, youden, auroc = loop.run_method(
            "GNG",
            splits,
            flag,
            auroc_prefix,
            actual_openness_percent,
            metrica="1_NN",
            epochs=int(job["epochs"]),
        )
    finally:
        gng.funcion_predecir_scoresREAL = original
    train_seconds = time.perf_counter() - t0

    if "pred" not in captured:
        raise RuntimeError("legacy run_method did not call funcion_predecir_scoresREAL")

    dist_mean, confidence, pred_base = _neighbor_stats(
        gng, captured["Xfeatures"], captured["Xlabels"], captured["test"], captured["scaler_mean"], captured["scaler_scale"]
    )
    # The scaler object is not kept. Rebuild the scaled matrices the same way _escalar does,
    # using the mean and scale captured from that scaler, via a tiny stand-in.
    pred_legacy = np.rint(np.asarray(captured["pred"], dtype=float)).astype(int)
    mask = reject_mask_1nn(
        dist_mean, confidence, captured["umbralDIST"], captured["umbralCONF"], captured["openess"]
    )
    pred_from_mask = pred_base.copy()
    pred_from_mask[mask] = 100
    mismatches = int(np.sum(pred_legacy != pred_from_mask))
    s = score_1nn(dist_mean, confidence, captured["umbralDIST"], captured["umbralCONF"])
    if captured["openess"] > 0:
        pred_from_score = pred_base.copy()
        pred_from_score[s > 1] = 100
        score_mismatches = int(np.sum(pred_legacy != pred_from_score))
    else:
        score_mismatches = int(np.sum(pred_legacy != pred_base))

    y_true_str, y_pred_str, y_closed_str = _stringify(gng, flag, splits, pred_legacy, pred_base)
    known = splits.kkc_labels_all
    paper_f1 = micro_f1_reference(y_true_str, y_pred_str, known)
    repo_f1 = micro_f1_repo(y_true_str, y_pred_str, known)
    parts = decompose(y_true_str, y_pred_str, y_closed_str, known)
    node_labels = np.rint(captured["Xlabels"].astype(float)).astype(int)
    _, counts = np.unique(node_labels, return_counts=True)

    record = {
        "ok": True,
        "dataset": job["dataset"],
        "partition_id": int(job["partition_id"]),
        "nominal_openness": float(job["nominal_openness"]),
        "actual_openness_percent": actual_openness_percent,
        "method": "1_NN",
        "epochs": int(job["epochs"]),
        "seed": int(job["seed"]),
        "config_hash": job["config_hash"],
        "known_classes": list(known),
        "fitting_classes": list(splits.kkc_labels_fit),
        "c_tr": int(part["c_tr"]),
        "c_te": int(part["c_te"]),
        "train_seconds": train_seconds,
        "repo_f1_percent_from_run_method": float(repo_f1_percent),
        "repo_f1_percent": float(100.0 * repo_f1),
        "paper_f1_percent": float(100.0 * paper_f1),
        "repo_f1_abs_diff_vs_run_method": float(abs(100.0 * repo_f1 - float(repo_f1_percent))),
        "accuracy_percent_from_run_method": float(acc_percent),
        "youden_from_run_method": float(youden),
        "auroc_from_run_method": None if auroc is None or (isinstance(auroc, float) and np.isnan(auroc)) else float(auroc),
        "threshold_dist": captured["umbralDIST"],
        "threshold_conf": captured["umbralCONF"],
        "decision_mismatches": mismatches,
        "score_gt1_mismatches": score_mismatches,
        "n_test": int(pred_legacy.size),
        "n_nodes": int(captured["Xfeatures"].shape[0]),
        "n_classes_with_fewer_than_5_nodes": int(np.sum(counts < 5)),
        "min_nodes_per_class": int(counts.min()) if counts.size else 0,
        "decomposition": parts,
    }
    _atomic_npz(
        Path(job["out_npz"]),
        dist_mean=dist_mean,
        confidence=confidence,
        score=s,
        pred_legacy=pred_legacy,
        pred_base=pred_base,
        y_true=np.array(y_true_str, dtype=object),
        y_pred=np.array(y_pred_str, dtype=object),
        node_positions=captured["Xfeatures"],
        node_labels=node_labels,
        scaler_mean=captured["scaler_mean"],
        scaler_scale=captured["scaler_scale"],
    )
    return record


def _neighbor_stats(gng, nodes, labels, test, mean, scale):
    """Repeat the scaled 5-NN geometry used inside ``funcion_predecir_scoresREAL``."""
    from sklearn.neighbors import KNeighborsClassifier

    class _Scaler:
        pass

    scaler = _Scaler()
    scaler.mean_ = np.asarray(mean, dtype=float)
    scaler.scale_ = np.asarray(scale, dtype=float)
    # StandardScaler.transform is (X - mean) / scale. _escalar calls scaler.transform.
    def _transform(matrix):
        return (np.asarray(matrix, dtype=float) - scaler.mean_) / scaler.scale_

    scaler.transform = _transform  # type: ignore[attr-defined]
    nodes_s, test_s = gng._escalar(np.asarray(nodes, dtype=float), np.asarray(test, dtype=float), scaler)
    knn = KNeighborsClassifier(n_neighbors=gng.K_VECINOS_OSR, algorithm="ball_tree")
    knn.fit(nodes_s, np.asarray(labels).ravel())
    distances, _ = knn.kneighbors(test_s)
    dist_mean = distances.mean(axis=1)
    pred_base = np.rint(knn.predict(test_s).astype(float)).astype(int)
    confidence = knn.predict_proba(test_s).max(axis=1)
    return dist_mean, confidence, pred_base


def _stringify(gng, flag: int, splits, pred: np.ndarray, pred_base: np.ndarray):
    if flag == 0:
        y_full = gng.target_to_numeric(splits.y_tr)
        y_test1 = gng.target_to_numeric(splits.y_te)
        y_test = [v if v in np.unique(y_full) else 100 for v in y_test1]
        y_true = gng.numeric_to_target(y_test)
        y_pred = gng.numeric_to_target([int(v) for v in pred])
        y_closed = gng.numeric_to_target([int(v) for v in pred_base])
    else:
        y_full = [int(v) for v in splits.y_tr]
        y_test1 = [100 if v == "UUC" else int(v) for v in splits.y_te]
        y_test = [v if v in np.unique(y_full) else 100 for v in y_test1]
        y_true = ["UUC" if v == 100 else str(v) for v in y_test]
        y_pred = ["UUC" if int(v) == 100 else str(int(v)) for v in pred]
        y_closed = ["UUC" if int(v) == 100 else str(int(v)) for v in pred_base]
    return y_true, y_pred, y_closed


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
