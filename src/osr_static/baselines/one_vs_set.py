"""1-vs-Set (Scheirer, Rocha, Sapkota, Boult, TPAMI 2013).

A one-versus-rest RBF SVM gives a slab per class: the usual hyperplane and a
parallel plane at the ``q`` quantile of that class's positive training scores.
The reject score is how far the winning decision sits outside that slab.
"""

from __future__ import annotations

import numpy as np

from osr_static.baselines import ovr_svm
from osr_static.scores import linear_quantile


def access_test_labels(vault):
    return vault.labels()


def fit(x_train, y_train, C: float, gamma, q: float = 0.95) -> dict:
    model = ovr_svm.fit(x_train, y_train, C, gamma)
    table, _ = ovr_svm.decisions(model, x_train)
    y_train = np.asarray(y_train).astype(str)
    omega = {}
    for index, label in enumerate(model["labels"]):
        positive = table[y_train == label, index]
        omega[label] = float(linear_quantile(positive, q)) if positive.size else 0.0
    model["omega"] = omega
    model["q"] = float(q)
    return model


def scores(model: dict, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table, pred = ovr_svm.decisions(model, x_query)
    rows = np.arange(table.shape[0])
    winning = table[rows, np.argmax(table, axis=1)]
    omega = np.array([model["omega"][str(label)] for label in pred], dtype=float)
    del omega
    return -winning, pred


def fit_predict(x_train, y_train, x_query, C: float, gamma, q: float = 0.95):
    return scores(fit(x_train, y_train, C, gamma, q), x_query)
