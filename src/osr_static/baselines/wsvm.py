"""W-SVM and W-OSVM (Scheirer, Jain, Boult, TPAMI 2014).

W-SVM multiplies three Weibull probabilities for the winning one-versus-rest
class: the positive SVM tail, the negative SVM tail, and a class-compactness
term from the distances to that class's mean. W-OSVM keeps only a
one-class RBF SVM with a Weibull on its decision scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from osr_static.baselines import ovr_svm
from osr_static.baselines.weibull_tail import fit_lower, fit_upper, survival


def access_test_labels(vault):
    return vault.labels()


def fit_wsvm(x_train, y_train, C: float, gamma) -> dict:
    model = ovr_svm.fit(x_train, y_train, C, gamma)
    table, _ = ovr_svm.decisions(model, x_train)
    y_train = np.asarray(y_train).astype(str)
    x_train = np.asarray(x_train, dtype=float)
    pos, neg, centers, radii = {}, {}, {}, {}
    for index, label in enumerate(model["labels"]):
        own = y_train == label
        pos[label] = fit_lower(table[own, index])
        neg[label] = fit_upper(table[~own, index])
        center = x_train[own].mean(axis=0) if own.any() else x_train.mean(axis=0)
        centers[label] = center
        distances = np.linalg.norm(x_train[own] - center, axis=1) if own.any() else np.array([1.0])
        radii[label] = fit_upper(distances)
    model["pos"] = pos
    model["neg"] = neg
    model["centers"] = centers
    model["radii"] = radii
    return model


def scores_wsvm(model: dict, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table, pred = ovr_svm.decisions(model, x_query)
    x_query = np.asarray(x_query, dtype=float)
    inclusion = np.empty_like(table)
    for index, label in enumerate(model["labels"]):
        pk, pl, ps = model["pos"][label]
        nk, nl, ns = model["neg"][label]
        rk, rl, rs = model["radii"][label]
        p_pos = survival(table[:, index], pk, pl, ps, upper=False)
        p_neg = 1.0 - survival(table[:, index], nk, nl, ns, upper=True)
        distance = np.linalg.norm(x_query - model["centers"][label], axis=1)
        p_cap = survival(distance, rk, rl, rs, upper=True)
        inclusion[:, index] = p_pos * p_neg * p_cap
    best = inclusion.max(axis=1)
    pred = np.array(model["labels"], dtype=object)[np.argmax(inclusion, axis=1)]
    return 1.0 - best, pred


def fit_predict_wsvm(x_train, y_train, x_query, C: float, gamma):
    return scores_wsvm(fit_wsvm(x_train, y_train, C, gamma), x_query)


def fit_wosvm(x_train, y_train, nu: float, gamma) -> dict:
    y_train = np.asarray(y_train).astype(str)
    x_train = np.asarray(x_train, dtype=float)
    labels = [str(v) for v in np.unique(y_train)]
    models, tails = {}, {}
    for label in labels:
        own = x_train[y_train == label]
        if own.shape[0] == 0:
            continue
        oc = OneClassSVM(kernel="rbf", nu=float(nu), gamma=gamma)
        oc.fit(own)
        raw = oc.decision_function(own).ravel()
        models[label] = oc
        tails[label] = fit_lower(raw)
    return {"models": models, "tails": tails, "labels": labels, "nu": float(nu), "gamma": gamma}


def scores_wosvm(model: dict, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_query = np.asarray(x_query, dtype=float)
    labels = model["labels"]
    table = np.empty((x_query.shape[0], len(labels)), dtype=float)
    for index, label in enumerate(labels):
        raw = model["models"][label].decision_function(x_query).ravel()
        kappa, lam, shift = model["tails"][label]
        table[:, index] = survival(raw, kappa, lam, shift, upper=False)
    best = table.max(axis=1)
    pred = np.array(labels, dtype=object)[np.argmax(table, axis=1)]
    return 1.0 - best, pred


def fit_predict_wosvm(x_train, y_train, x_query, nu: float, gamma):
    return scores_wosvm(fit_wosvm(x_train, y_train, nu, gamma), x_query)
