"""Probability-of-inclusion SVM (Jain, Scheirer, Boult, ECCV 2014).

A Weibull is fit to the lower tail of each class's positive one-versus-rest
scores. The inclusion probability is that Weibull's complementary CDF. The
reject score is one minus the largest inclusion probability.
"""

from __future__ import annotations

import numpy as np

from osr_static.baselines import ovr_svm
from osr_static.baselines.weibull_tail import fit_lower, survival


def access_test_labels(vault):
    return vault.labels()


def fit(x_train, y_train, C: float, gamma) -> dict:
    model = ovr_svm.fit(x_train, y_train, C, gamma)
    table, _ = ovr_svm.decisions(model, x_train)
    y_train = np.asarray(y_train).astype(str)
    tails = {}
    for index, label in enumerate(model["labels"]):
        positive = table[y_train == label, index]
        tails[label] = fit_lower(positive)
    model["tails"] = tails
    return model


def scores(model: dict, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table, pred = ovr_svm.decisions(model, x_query)
    inclusion = np.empty_like(table)
    for index, label in enumerate(model["labels"]):
        kappa, lam, shift = model["tails"][label]
        inclusion[:, index] = survival(table[:, index], kappa, lam, shift, upper=False)
    best = inclusion.max(axis=1)
    pred = np.array(model["labels"], dtype=object)[np.argmax(inclusion, axis=1)]
    return 1.0 - best, pred


def fit_predict(x_train, y_train, x_query, C: float, gamma):
    return scores(fit(x_train, y_train, C, gamma), x_query)
