"""Persist the legacy class lists and the seed-123 sample indices.

Index construction follows ``create_splits_explicit`` in the legacy script:
``np.random.default_rng(123)``, ``ceil(60%)`` train per known class, fitting
classes are the first ``int(2/3 * |KKC|)`` labels in log order, then another
``ceil(60%)`` of those training rows. Unknown test rows and simulated-unknown
validation rows are labeled ``UUC`` in the materialized splits.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from osr_static.data import load_txt_any

KNOWN_SPLIT = "known_class"


def read_partition_log(log_path: str | Path) -> list[list[int]]:
    """Same reader the legacy main loop uses (substring split, not the bracket parser)."""
    known_lists: list[list[int]] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if KNOWN_SPLIT in line:
                nums = re.findall(r"\d+", line.split(KNOWN_SPLIT)[1])
                known_lists.append([int(x) for x in nums])
    return known_lists


def normalize_known_for_dataset(dataset_name: str, known_ids: Sequence[int], y_all: np.ndarray) -> list[str]:
    """Map log ids onto the label strings stored in the table."""
    uniq = np.unique(y_all)
    if dataset_name.lower() == "letter":
        if all((len(str(u)) == 1 and str(u).isalpha()) for u in uniq):
            return [chr(64 + int(k)) for k in known_ids]
        return [str(int(k)) for k in known_ids]
    return [str(int(k)) for k in known_ids]


def openness(c_tr: int, c_te: int) -> float:
    """``1 - sqrt(2 |C_tr| / (|C_tr| + |C_te|))``, as a fraction."""
    return 1.0 - math.sqrt((2 * c_tr) / (c_tr + c_te))


def _split_indices(x: np.ndarray, y: np.ndarray, known_labels: Sequence[str]) -> dict[str, Any]:
    known_labels = [str(k) for k in known_labels]
    all_labels = [str(v) for v in np.unique(y).tolist()]
    unknown_labels = [c for c in all_labels if c not in known_labels]
    y = np.asarray(y).astype(str)

    idx_kkc = np.flatnonzero(np.isin(y, known_labels))
    idx_uuc = np.flatnonzero(~np.isin(y, known_labels))
    y_kkc = y[idx_kkc]
    rng = np.random.default_rng(123)

    train_parts: list[np.ndarray] = []
    test_known_parts: list[np.ndarray] = []
    for label in known_labels:
        local = np.where(y_kkc == label)[0]
        local = rng.permutation(local)
        n_train = math.ceil(0.6 * len(local))
        train_parts.append(idx_kkc[local[:n_train]])
        test_known_parts.append(idx_kkc[local[n_train:]])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_known_parts + [idx_uuc])

    n_fit = int(2 / 3 * len(known_labels))
    fitting = known_labels[:n_fit]
    simulated = known_labels[n_fit:]
    y_train = y[train_idx]
    fit_rows = np.flatnonzero(np.isin(y_train, fitting))
    sim_rows = np.flatnonzero(~np.isin(y_train, fitting))
    idx_tf = train_idx[fit_rows]
    idx_tu = train_idx[sim_rows]
    y_tf = y[idx_tf]

    fit_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in fitting:
        local = np.where(y_tf == label)[0]
        local = rng.permutation(local)
        n_fit_rows = math.ceil(0.6 * len(local))
        fit_parts.append(idx_tf[local[:n_fit_rows]])
        val_parts.append(idx_tf[local[n_fit_rows:]])

    fit_idx = np.concatenate(fit_parts)
    val_closed_idx = np.concatenate(val_parts)
    val_open_idx = np.concatenate([val_closed_idx, idx_tu])
    return {
        "known_classes": known_labels,
        "unknown_classes": unknown_labels,
        "fitting_classes": fitting,
        "simulated_uuc_classes": simulated,
        "c_tr": len(known_labels),
        "c_te": len(all_labels),
        "actual_openness": openness(len(known_labels), len(all_labels)),
        "indices": {
            "train": train_idx.astype(int).tolist(),
            "fit": fit_idx.astype(int).tolist(),
            "val_closed": val_closed_idx.astype(int).tolist(),
            "val_open": val_open_idx.astype(int).tolist(),
            "val_simulated": idx_tu.astype(int).tolist(),
            "test": test_idx.astype(int).tolist(),
        },
    }


def materialize(x: np.ndarray, y: np.ndarray, record: Mapping[str, Any]) -> dict[str, Any]:
    """Slice ``X`` and ``y`` and apply the legacy ``UUC`` relabeling."""
    y = np.asarray(y).astype(str)
    idx = record["indices"]
    test_idx = np.asarray(idx["test"], dtype=int)
    val_open_idx = np.asarray(idx["val_open"], dtype=int)
    val_sim = set(int(i) for i in idx["val_simulated"])
    known = set(record["known_classes"])

    def take(name: str) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(idx[name], dtype=int)
        return x[rows], y[rows]

    x_fit, y_fit = take("fit")
    x_val_c, y_val_c = take("val_closed")
    x_tr, y_tr = take("train")
    x_te = x[test_idx]
    y_te = np.array(["UUC" if str(y[i]) not in known else str(y[i]) for i in test_idx], dtype=object)
    x_vo = x[val_open_idx]
    y_vo = np.array(["UUC" if int(i) in val_sim else str(y[i]) for i in val_open_idx], dtype=object)
    return {
        "X_fit": x_fit,
        "y_fit": y_fit,
        "X_val_closed": x_val_c,
        "y_val_closed": y_val_c,
        "X_val_open": x_vo,
        "y_val_open": y_vo,
        "X_te": x_te,
        "y_te": y_te,
        "X_tr": x_tr,
        "y_tr": y_tr,
        "kkc_labels_fit": list(record["fitting_classes"]),
        "kkc_labels_all": list(record["known_classes"]),
    }


def build_dataset_document(
    dataset: str,
    x: np.ndarray,
    y: np.ndarray,
    logs: Mapping[str, Path],
) -> dict[str, Any]:
    """One JSON document: every nominal openness and every log line."""
    partitions: list[dict[str, Any]] = []
    for nominal, log_path in logs.items():
        known_lists = read_partition_log(log_path)
        for partition_id, known_ids in enumerate(known_lists, start=1):
            labels = normalize_known_for_dataset(dataset, known_ids, y)
            block = _split_indices(x, y, labels)
            block["partition_id"] = partition_id
            block["nominal_openness"] = float(nominal)
            block["log_file"] = Path(log_path).name
            block["known_ids"] = [int(v) for v in known_ids]
            partitions.append(block)
    return {
        "dataset": dataset,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "split_seed": 123,
        "partitions": partitions,
    }


def write_partitions(data_dir: Path, out_dir: Path, datasets: Mapping[str, Mapping[str, str]]) -> None:
    """Build ``<dataset>.json`` for each dataset described in the config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, spec in datasets.items():
        x, y = load_txt_any(data_dir / spec["file"])
        logs = {str(key): data_dir / name for key, name in spec["logs"].items()}
        document = build_dataset_document(dataset, x, y, logs)
        target = out_dir / f"{dataset}.json"
        _atomic_json(target, document)


def load_partitions(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_record(document: Mapping[str, Any], partition_id: int, nominal_openness: float) -> dict[str, Any]:
    for record in document["partitions"]:
        if int(record["partition_id"]) == int(partition_id) and abs(
            float(record["nominal_openness"]) - float(nominal_openness)
        ) < 1e-9:
            return record
    raise KeyError(f"partition {partition_id} openness {nominal_openness} not in {document.get('dataset')}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    temporary.replace(path)
