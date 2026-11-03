"""Historical note for the libsvm-openset binary.

W-SVM, W-OSVM, and PI-SVM now live in ``wsvm.py`` and ``pisvm.py`` as
scikit-learn one-versus-rest models. This module stays so older JSON still
has a field to read.
"""

from __future__ import annotations


def access_test_labels(vault):
    return vault.labels()


def available() -> tuple[bool, str]:
    return True, "sklearn W-SVM / PI-SVM / 1-vs-Set; libsvm-openset is not required"
