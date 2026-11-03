"""Prequential stream evaluation for batch novelty detectors (Table I protocol)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from osr_static.baselines import knnens, minas, sencforest
from osr_static.baselines.stream_common import map_novelty_predictions
from osr_static.baseline_runner import _knnens_select, _minas_select, _senc_select
from osr_static.metrics import apply_threshold
from osr_static.stream_benchmark import batch_slices, layout_for
from osr_static.synthetic import load_synthetic
from osr_static.synthetic_metrics import numeric_predictions, numeric_test_labels

REJECT_NUM = 100


class _OSRMetricsLite:
    @staticmethod
    def micro_f1(y_true, y_pred, known_labels) -> float:
        known = [str(k) for k in known_labels]
        yt = np.asarray(y_true, dtype=str)
        yp = np.asarray(y_pred, dtype=str)
        known_set = set(known)
        tp = fp = fn = 0
        for t, p in zip(yt, yp):
            tk = t in known_set
            if tk and t == p:
                tp += 1
            elif tk and t != p:
                fn += 1
            elif not tk and p in known_set:
                fp += 1
        den = 2 * tp + fp + fn
        return (2 * tp / den) if den > 0 else 0.0


def _hyperparam_select(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    ts: int,
    known: list[str],
) -> tuple[dict, float | None]:
    """Tune on the supervised prefix only (``t < ts``)."""
    x_s = x[:ts]
    y_s = np.asarray(y[:ts]).astype(str)
    known_fit = known[: max(1, (2 * len(known)) // 3)]
    if len(known_fit) < 1:
        known_fit = list(known)
    rng = np.random.default_rng(123)
    idx = rng.permutation(ts)
    n_val = max(1, ts // 5)
    val_idx = idx[:n_val]
    fit_idx = idx[n_val:]
    x_fit, y_fit = x_s[fit_idx], y_s[fit_idx]
    x_val, y_val = x_s[val_idx], y_s[val_idx]
    rejection_off = False
    if method == "minas":
        params, tau = _minas_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off)
        return params, 1.0
    if method == "knnens":
        params, _ = _knnens_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off)
        return params, None
    if method == "sencforest":
        params, _ = _senc_select(x_fit, y_fit, x_val, y_val, known_fit, rejection_off)
        return params, None
    raise ValueError(method)


def _fit_model(method: str, params: dict, x_tr: np.ndarray, y_tr: np.ndarray):
    y_tr = np.asarray(y_tr).astype(str)
    if method == "minas":
        return minas.fit(x_tr, y_tr, params["k_micro"], params["radius_factor"])
    if method == "knnens":
        return knnens.fit(x_tr, y_tr, params["b"], params["a"], params["L"])
    if method == "sencforest":
        return sencforest.fit(
            x_tr, y_tr, params["n_estimators"], params["max_samples"], params["min_size"]
        )
    raise ValueError(method)


def _predict(
    method: str,
    model: Any,
    params: dict,
    tau: float | None,
    x_te: np.ndarray,
) -> np.ndarray:
    if method == "minas":
        scores, pred = minas.scores(model, x_te)
        pred = apply_threshold(scores, pred, tau if tau is not None else 1.0)
        return np.asarray(pred, dtype=object)
    if method == "knnens":
        t_apply = params["t"]
        _, pred = knnens.scores(model, x_te, t_apply)
        return map_novelty_predictions(pred)
    if method == "sencforest":
        _, pred = sencforest.scores(model, x_te)
        return map_novelty_predictions(pred)
    raise ValueError(method)


def _update_minas(model: dict, x_new: np.ndarray, k_extra: int = 2) -> dict:
    """Unlabeled batch: add micro-clusters (novelty shells) from features only."""
    if x_new.shape[0] == 0:
        return model
    from sklearn.cluster import KMeans

    k = max(1, min(int(k_extra), x_new.shape[0]))
    km = KMeans(n_clusters=k, n_init=5, random_state=0)
    km.fit(x_new)
    cents = list(model["centroids"])
    rads = list(model["radii"])
    labs = list(model["labels"])
    rf = float(model["radius_factor"])
    for c in range(k):
        mask = km.labels_ == c
        pts = x_new[mask] if np.any(mask) else x_new
        center = km.cluster_centers_[c]
        dist = np.linalg.norm(pts - center, axis=1)
        spread = float(dist.std(ddof=0)) if dist.size > 1 else float(max(dist.max(), 1e-6))
        cents.append(center)
        rads.append(rf * max(spread, 1e-6))
        labs.append(str(REJECT_NUM))
    out = dict(model)
    out["centroids"] = np.asarray(cents, dtype=float)
    out["radii"] = np.asarray(rads, dtype=float)
    out["labels"] = np.asarray(labs, dtype=object)
    return out


def evaluate_stream_paper(
    name: str,
    method: str,
) -> dict[str, Any]:
    x, y_raw, _known_hint = load_synthetic(name)
    y = np.asarray(y_raw).astype(str)
    stem = name.replace(".txt", "")
    layout = layout_for(x.shape[0], stem)
    ts = layout.ts

    classes_sup = sorted({str(v) for i, v in enumerate(y) if i < ts})
    known = classes_sup

    scaler = StandardScaler().fit(x[:ts])
    x = scaler.transform(x)

    params, tau = _hyperparam_select(method, x, y, ts, known)
    model = _fit_model(method, params, x[:ts], y[:ts])

    y_true_all: list[str] = []
    y_pred_all: list[object] = []
    y_tr_hist = y[:ts]
    errors: list[float] = []

    for train_idx, test_idx in batch_slices(layout):
        x_te = x[test_idx]
        y_te = y[test_idx]
        pred = _predict(method, model, params, tau, x_te)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(pred.tolist())

        yt_num = numeric_test_labels(y_te, y_tr_hist)
        yp_num = numeric_predictions(pred)
        errors.extend((yt_num != yp_num).astype(float).tolist())

        lab_idx = train_idx[train_idx < ts]
        unlab_idx = train_idx[train_idx >= ts]
        if lab_idx.size > 0:
            model = _fit_model(method, params, x[lab_idx], y[lab_idx])
        if unlab_idx.size > 0 and method == "minas":
            model = _update_minas(model, x[unlab_idx])

    preq = float(100.0 * np.mean(errors)) if errors else float("nan")
    metrics = _OSRMetricsLite()
    f1 = float(100.0 * metrics.micro_f1(y_true_all, y_pred_all, known))

    return {
        "ok": True,
        "dataset": stem,
        "file": name,
        "method": method,
        "protocol": "table_i_stream",
        "n": layout.n,
        "ts": ts,
        "vol_batch": layout.vol_batch,
        "n_stream_steps": layout.n_steps,
        "known_supervised_classes": known,
        "selected_params": params,
        "threshold": tau,
        "paper_f1_percent": f1,
        "prequential_error_percent": preq,
        "n_scored": len(errors),
    }
