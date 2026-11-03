"""Environment stamp for a run. There is no git repository in this tree."""

from __future__ import annotations

import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from osr_static.jobs import atomic_json
from osr_static.paths import DATA_STATIC, REPO_ROOT, RESULTS_STATIC


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(stage: str) -> Path:
    repo = REPO_ROOT
    versions = {}
    for name in ("numpy", "pandas", "sklearn", "yaml", "matplotlib", "scipy", "pytest"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    cpu = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    data = {}
    for name in ("letter_dataset.csv", "pendigits_dataset.csv"):
        path = DATA_STATIC / name
        if path.exists():
            data[name] = _sha256(path)
    payload = {
        "stage": stage,
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": None,
        "git_status": "this tree is not a git repository",
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu": cpu,
        "cpu_count": __import__("os").cpu_count(),
        "packages": versions,
        "data_sha256": data,
    }
    dest = RESULTS_STATIC / "manifest.json"
    atomic_json(dest, payload)
    config_dir = RESULTS_STATIC / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for path in (repo / "configs").glob("*.yaml"):
        (config_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="start")
    path = write_manifest(parser.parse_args().stage)
    print(path)


if __name__ == "__main__":
    main()
