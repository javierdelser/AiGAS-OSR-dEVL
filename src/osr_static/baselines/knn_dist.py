"""k-NN whose score is the distance to the k-th training neighbor."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def access_test_labels(vault):
    return vault.labels()


def scores(x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train).astype(str)
    k_use = min(int(k), x_train.shape[0])
    model = KNeighborsClassifier(n_neighbors=k_use, algorithm="ball_tree")
    model.fit(x_train, y_train)
    dist, _ = model.kneighbors(x_query, n_neighbors=k_use)
    return dist[:, -1], model.predict(x_query)
