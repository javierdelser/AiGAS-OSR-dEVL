"""Micro-F1, the paper definition and the repository definition, plus the error split.

``micro_f1_reference`` counts a false positive for every sample whose true label
is not class ``i`` and whose prediction is ``i``, including other known classes
and unknowns. That is the Mendes Júnior micro-F1 used in the paper.

``micro_f1_repo`` counts a false positive only when an unknown sample is
predicted as a known class. That is ``OSRMetrics.micro_f1``. A confusion
between two known classes is only a false negative there, so the two numbers
differ as soon as that happens.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


REJECT_LABEL = "UUC"


def normalize_label(value: object) -> str:
    """Same numeric canonicalisation as ``OSRMetrics._unify_types``."""
    if value is None:
        return "None"
    text = str(value).strip()
    if text == "":
        return ""
    try:
        as_float = float(text)
    except (TypeError, ValueError):
        return text
    if as_float == int(as_float):
        return str(int(as_float))
    return str(as_float)


def _as_labels(values: Iterable[object]) -> np.ndarray:
    return np.array([normalize_label(v) for v in np.asarray(values).ravel()], dtype=object)


def micro_f1_reference(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    known_classes: Sequence[object],
    reject_label: str = REJECT_LABEL,
) -> float:
    """Paper micro-F1. ``reject_label`` is not a class; a rejected known sample is a false negative.

    ``P = ΣTP / Σ(TP+FP)``, ``R = ΣTP / Σ(TP+FN)``, ``F1 = 2PR/(P+R)``.
    ``F1 = 0`` when ``P + R = 0``.
    """
    del reject_label  # documented; rejection is just any prediction that is not the true class
    yt = _as_labels(y_true)
    yp = _as_labels(y_pred)
    known = [normalize_label(k) for k in known_classes]
    tp = fp = fn = 0
    for label in known:
        tp += int(np.sum((yt == label) & (yp == label)))
        fp += int(np.sum((yt != label) & (yp == label)))
        fn += int(np.sum((yt == label) & (yp != label)))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def micro_f1_repo(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    known_classes: Sequence[object],
    reject_label: str = REJECT_LABEL,
) -> float:
    """Repository micro-F1: false positives are unknowns predicted as a known class only."""
    del reject_label
    yt = _as_labels(y_true)
    yp = _as_labels(y_pred)
    known = np.array([normalize_label(k) for k in known_classes], dtype=object)
    mask_known = np.isin(yt, known)
    tp = int(np.sum((yt == yp) & mask_known))
    fn = int(np.sum((yt != yp) & mask_known))
    fp = int(np.sum(np.isin(yp, known) & ~mask_known))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def micro_f1_sweep(
    scores: np.ndarray,
    pred_labels: Sequence[object],
    y_true: Sequence[object],
    known_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper micro-F1 at every distinct threshold, including reject-all and accept-all.

    ``scores`` are higher for more unknown. A sample is accepted when
    ``score <= tau``. ``pred_labels`` are the classifier labels before rejection.
    Ties stay together. The returned thresholds are ``-inf`` (reject all) and
    then each distinct score (accept every sample with score less than or equal
    to it). Accept-all is the last entry, because every score is ``<= max(score)``.
    """
    scores = np.asarray(scores, dtype=float)
    pred = _as_labels(pred_labels)
    true = _as_labels(y_true)
    known_mask = np.asarray(known_mask, dtype=bool)
    n_known = int(known_mask.sum())
    if scores.size == 0:
        return np.array([0.0]), np.array([-np.inf])

    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    correct = known_mask[order] & (pred[order] == true[order])
    cumulative_tp = np.cumsum(correct.astype(int))

    f1 = [0.0]
    tau = [-np.inf]
    start = 0
    n = ordered_scores.size
    while start < n:
        end = start
        while end + 1 < n and ordered_scores[end + 1] == ordered_scores[start]:
            end += 1
        tp = int(cumulative_tp[end])
        accepted = end + 1
        f1.append(_f1_from_parts(tp, accepted, n_known))
        tau.append(float(ordered_scores[start]))
        start = end + 1
    return np.asarray(f1, dtype=float), np.asarray(tau, dtype=float)


def _f1_from_parts(tp: int, accepted: int, n_known: int) -> float:
    if accepted == 0 or n_known == 0:
        return 0.0
    precision = tp / accepted
    recall = tp / n_known
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def apply_threshold(
    scores: np.ndarray,
    pred_labels: Sequence[object],
    tau: float,
    reject_label: str = REJECT_LABEL,
) -> np.ndarray:
    """Accept when ``score <= tau``; otherwise emit ``reject_label``."""
    pred = _as_labels(pred_labels)
    rejected = np.asarray(scores, dtype=float) > tau
    out = pred.copy()
    out[rejected] = normalize_label(reject_label)
    return out


def decompose(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    y_pred_closed: Sequence[object],
    known_classes: Sequence[object],
    reject_label: str = REJECT_LABEL,
) -> dict[str, float]:
    """Operating-point split from the prompt, using the paper's TP/FP/FN.

    ``y_pred_closed`` is the classifier label with rejection switched off.
    """
    yt = _as_labels(y_true)
    yp = _as_labels(y_pred)
    yc = _as_labels(y_pred_closed)
    known = np.array([normalize_label(k) for k in known_classes], dtype=object)
    reject = normalize_label(reject_label)
    known_mask = np.isin(yt, known)
    unknown_mask = ~known_mask
    n_known = int(known_mask.sum())
    n_unknown = int(unknown_mask.sum())

    correct_closed = known_mask & (yc == yt)
    rejected = known_mask & (yp == reject)
    accepted_known = known_mask & (yp != reject)
    tp_mask = known_mask & (yp == yt)
    fn_mis = accepted_known & (yp != yt)
    fp_known = accepted_known & (yp != yt)
    fp_unknown = unknown_mask & np.isin(yp, known)

    tp = int(tp_mask.sum())
    fn_rejected = int(rejected.sum())
    fn_misclassified = int(fn_mis.sum())
    fp_from_known = int(fp_known.sum())
    fp_from_unknown = int(fp_unknown.sum())

    closed_acc = float(correct_closed.sum() / n_known) if n_known else 0.0
    frr = float(fn_rejected / n_known) if n_known else 0.0
    far = float((unknown_mask & (yp != reject)).sum() / n_unknown) if n_unknown else 0.0
    n_accepted_known = tp + fn_misclassified
    acc_accepted = float(tp / n_accepted_known) if n_accepted_known else 0.0
    recall_from_frr = (1.0 - frr) * acc_accepted
    fp_total = fp_from_known + fp_from_unknown
    precision = float(tp / (tp + fp_total)) if (tp + fp_total) else 0.0
    if precision + recall_from_frr == 0:
        f1_predicted = 0.0
    else:
        f1_predicted = float(2 * precision * recall_from_frr / (precision + recall_from_frr))

    return {
        "closed_acc": closed_acc,
        "FRR": frr,
        "FAR": far,
        "TP": tp,
        "FP_misclassified_known": fp_from_known,
        "FP_accepted_unknown": fp_from_unknown,
        "FN_rejected_known": fn_rejected,
        "FN_misclassified_known": fn_misclassified,
        "acc_on_accepted_known": acc_accepted,
        "F1_predicted": f1_predicted,
        "n_known": n_known,
        "n_unknown": n_unknown,
    }


def auroc_unknown(scores: np.ndarray, unknown_mask: np.ndarray) -> float:
    """AUROC with the unknown class as the positive class. ``scores`` are higher for more unknown."""
    from sklearn.metrics import roc_auc_score

    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    if unknown_mask.sum() == 0 or (~unknown_mask).sum() == 0:
        return float("nan")
    return float(roc_auc_score(unknown_mask.astype(int), np.asarray(scores, dtype=float)))


def aupr_unknown(scores: np.ndarray, unknown_mask: np.ndarray) -> float:
    """Average precision with the unknown class as the positive class."""
    from sklearn.metrics import average_precision_score

    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    if unknown_mask.sum() == 0 or (~unknown_mask).sum() == 0:
        return float("nan")
    return float(average_precision_score(unknown_mask.astype(int), np.asarray(scores, dtype=float)))


def oscr_curve(
    scores: np.ndarray,
    correct_known: np.ndarray,
    unknown_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """CCR versus FPR. Accept when ``score <= tau``. Area is trapezoidal on FPR in ``[0, 1]``.

    ``correct_known`` is true only for known samples whose closed-set label is correct.
    Both endpoints are included: reject-all is ``(FPR=0, CCR=0)``, accept-all is
    ``(FPR=1, CCR=closed-set accuracy)`` when every unknown can be accepted.
    """
    scores = np.asarray(scores, dtype=float)
    correct_known = np.asarray(correct_known, dtype=bool)
    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    n_known = int((~unknown_mask).sum())
    n_unknown = int(unknown_mask.sum())
    if n_known == 0 or n_unknown == 0 or scores.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), float("nan")

    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    correct = np.cumsum(correct_known[order].astype(int))
    accepted_unknown = np.cumsum(unknown_mask[order].astype(int))
    fpr = [0.0]
    ccr = [0.0]
    start = 0
    n = ordered.size
    while start < n:
        end = start
        while end + 1 < n and ordered[end + 1] == ordered[start]:
            end += 1
        fpr.append(float(accepted_unknown[end] / n_unknown))
        ccr.append(float(correct[end] / n_known))
        start = end + 1
    if fpr[-1] != 1.0:
        fpr.append(1.0)
        ccr.append(ccr[-1])
    fpr_arr = np.asarray(fpr, dtype=float)
    ccr_arr = np.asarray(ccr, dtype=float)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return fpr_arr, ccr_arr, float(trap(ccr_arr, fpr_arr))


def downsample_curve(x: np.ndarray, y: np.ndarray, limit: int = 500) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= limit:
        return x, y
    idx = np.unique(np.linspace(0, x.size - 1, limit).astype(int))
    return x[idx], y[idx]


def operating_report(vault, scores: np.ndarray, pred_closed, tau: float | None) -> dict:
    """Paper F1, threshold-free metrics and the error split. Opens the label vault."""
    y_true = vault.labels()
    known_mask = vault.known_mask()
    known = [str(v) for v in vault._known.tolist()]
    pred_closed = _as_labels(pred_closed)
    if tau is None:
        y_pred = pred_closed
    else:
        y_pred = apply_threshold(scores, pred_closed, tau)
    unknown = ~known_mask
    correct_known = known_mask & (pred_closed == y_true)
    fpr, ccr, area = oscr_curve(scores, correct_known, unknown)
    f1 = micro_f1_reference(y_true, y_pred, known)
    parts = decompose(y_true, y_pred, pred_closed, known)
    return {
        "paper_f1": f1,
        "paper_f1_percent": float(100.0 * f1),
        "repo_f1_percent": float(100.0 * micro_f1_repo(y_true, y_pred, known)),
        "auroc": auroc_unknown(scores, unknown),
        "aupr": aupr_unknown(scores, unknown),
        "oscr": area,
        "oscr_fpr": fpr,
        "oscr_ccr": ccr,
        "decomposition": parts,
        "y_pred": y_pred,
    }
