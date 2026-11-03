"""One-versus-rest RBF SVMs. Shared by 1-vs-set, W-SVM, and PI-SVM."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def access_test_labels(vault):
    return vault.labels()


def fit(x_train: np.ndarray, y_train: np.ndarray, C: float, gamma) -> dict:
    """One binary RBF SVM per class. ``probability`` stays off."""
    y_train = np.asarray(y_train).astype(str)
    labels = np.unique(y_train)
    models = {}
    for label in labels:
        binary = np.where(y_train == label, 1, -1)
        model = SVC(C=float(C), kernel="rbf", gamma=gamma, class_weight="balanced")
        model.fit(x_train, binary)
        models[str(label)] = model
    return {"models": models, "labels": [str(v) for v in labels], "C": float(C), "gamma": gamma}


def decisions(model: dict, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n, n_classes)`` decision values and the class of the largest one."""
    x_query = np.asarray(x_query, dtype=float)
    labels = model["labels"]
    table = np.column_stack([model["models"][label].decision_function(x_query) for label in labels])
    chosen = np.argmax(table, axis=1)
    pred = np.array(labels, dtype=object)[chosen]
    return table, pred
