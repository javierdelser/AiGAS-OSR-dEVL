"""Job ids and the deterministic seed used before the legacy GNG call.

The legacy script does not seed NumPy. NeuPy shuffles each epoch with
``np.random.shuffle``, and the silhouette ``KMeans`` calls inside
``run_method`` also draw from that global state. Seeding once per job, from
the key below, makes a job reproducible on its own. It does not reproduce an
unseeded multi-job run of ``run.sh``, because that run lets the global state
carry from one partition into the next.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def config_hash(payload: Mapping[str, Any]) -> str:
    """Stable hash of the experiment settings that affect a result."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def derive_seed(
    dataset: str,
    partition_id: int,
    nominal_openness: float,
    method: str,
    config_hash_value: str,
) -> int:
    """Seed in ``[0, 2**32)`` from ``(dataset, partition_id, openness, method, config_hash)``.

    ``nominal_openness`` is formatted with four decimal places so ``0`` and
    ``0.0`` share a seed. This is the only seed function used by the runner.
    """
    text = f"{dataset}|{int(partition_id)}|{float(nominal_openness):.4f}|{method}|{config_hash_value}"
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:4], "little")


def experiment_root():
    """Repository root (``Release/``)."""
    from osr_static.paths import REPO_ROOT

    return REPO_ROOT


def partition_file(dataset: str):
    """Partition JSON under ``data/osr_static/partitions/``."""
    from osr_static.paths import DATA_STATIC, REPO_ROOT

    name = f"{dataset}.json"
    for path in (
        DATA_STATIC / "partitions" / name,
        REPO_ROOT / "data" / "partitions" / name,
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No partition file for {dataset} under data/osr_static/partitions/.")


def dataset_file(name: str):
    """CSV or partition log under ``data/osr_static/``."""
    from osr_static.paths import DATA_STATIC

    path = DATA_STATIC / name
    if path.is_file():
        return path
    raise FileNotFoundError(f"Could not find {name} under {DATA_STATIC}.")


def atomic_json(path, payload: Mapping[str, Any]) -> None:
    """Write JSON via a temporary file and ``replace``."""
    from pathlib import Path
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2)
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path, **arrays) -> None:
    """Write a compressed npz via a temporary file and ``replace``."""
    import os
    from pathlib import Path

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
