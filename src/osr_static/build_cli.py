"""Build partition files with the garage interpreter and check them against the legacy splitter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _same_rows(left: np.ndarray, right: np.ndarray, name: str) -> None:
    if left.shape != right.shape:
        raise SystemExit(f"{name} shape {left.shape} != {right.shape}")
    if left.dtype.kind == "f":
        if not np.allclose(left, right):
            raise SystemExit(f"{name} values differ")
        return
    if list(map(str, np.asarray(left).ravel())) != list(map(str, np.asarray(right).ravel())):
        raise SystemExit(f"{name} labels differ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    repo_root = Path(request["repo_root"])
    sys.path.insert(0, str(repo_root))
    sys.argv = [sys.argv[0]]

    import osr_GNG_Train_VAl_Test_LOOP_Split26 as loop
    from osr_static.data import load_txt_any
    from osr_static.partitions import materialize, read_partition_log, write_partitions

    data_dir = Path(request["data_dir"])
    out_dir = Path(request["out_dir"])
    for dataset, spec in request["datasets"].items():
        path = data_dir / spec["file"]
        x, y = load_txt_any(path)
        x_legacy, y_legacy = loop.load_txt_any(str(path))
        _same_rows(x, x_legacy, f"{dataset} X")
        _same_rows(y, y_legacy, f"{dataset} y")
        for nominal, log_name in spec["logs"].items():
            ours = read_partition_log(data_dir / log_name)
            theirs = loop.read_partition_log(str(data_dir / log_name))
            if ours != theirs:
                raise SystemExit(f"{log_name} readers disagree")

    write_partitions(data_dir, out_dir, request["datasets"])

    for dataset, spec in request["datasets"].items():
        path = data_dir / spec["file"]
        x, y = load_txt_any(path)
        document = json.loads((out_dir / f"{dataset}.json").read_text(encoding="utf-8"))
        for record in document["partitions"]:
            arrays = materialize(x, y, record)
            legacy = loop.create_splits_explicit(x, y, record["known_classes"])
            for field in (
                "X_fit", "y_fit", "X_val_closed", "y_val_closed", "X_val_open", "y_val_open",
                "X_te", "y_te", "X_tr", "y_tr",
            ):
                _same_rows(arrays[field], getattr(legacy, field), f"{dataset} p{record['partition_id']} {field}")
            if list(arrays["kkc_labels_fit"]) != list(legacy.kkc_labels_fit):
                raise SystemExit("fitting classes differ")
            if list(arrays["kkc_labels_all"]) != list(legacy.kkc_labels_all):
                raise SystemExit("known classes differ")
        print(f"checked {dataset}: {len(document['partitions'])} partitions")


if __name__ == "__main__":
    main()
