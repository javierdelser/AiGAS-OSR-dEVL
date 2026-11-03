"""One holdout GNG job. Invoked with the garage Python 3.8 interpreter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    from osr_static.holdout_train import execute_job

    execute_job(job)


if __name__ == "__main__":
    main()
