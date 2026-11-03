"""RBF SVM with a probability reject score ``1 - max class probability``."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def access_test_labels(vault):
    return vault.labels()


def fit_predict(x_train, y_train, x_query, C: float, gamma) -> tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train).astype(str)
    model = SVC(kernel="rbf", C=float(C), gamma=gamma, probability=True)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_query)
    score = 1.0 - proba.max(axis=1)
    pred = model.classes_[np.argmax(proba, axis=1)]
    return score, pred
