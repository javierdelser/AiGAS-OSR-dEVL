#!/usr/bin/env bash
# Minimal Release/: stream tables only; one run_benchmark.py per protocol.
set -eu
set -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REL="$ROOT/Release"
EXP="$ROOT/experiments/osr_static"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Building minimal Release from $ROOT → $REL"
T="$STAGE/tree"

mkdir -p \
  "$T/data/osr_static/partitions" \
  "$T/data/osr_dynamic" \
  "$T/scripts/osr_static" \
  "$T/src" \
  "$T/configs" \
  "$T/results/osr_static" \
  "$T/results/osr_dynamic" \
  "$T/docs"

rsync -a "$ROOT/DATA/" "$T/data/osr_static/"
rsync -a "$EXP/data/partitions/" "$T/data/osr_static/partitions/"
rsync -a "$EXP/data/osr_synthetic/datasets/" "$T/data/osr_dynamic/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' "$EXP/src/" "$T/src/"
rsync -a "$REL/src/osr_static/" "$T/src/osr_static/"
cp -a "$EXP/configs/base.yaml" "$EXP/configs/baselines_stream.yaml" "$T/configs/"
rsync -a "$EXP/requirements.lock.txt" "$T/" 2>/dev/null || cp "$ROOT/requirements.txt" "$T/requirements.lock.txt"

cp -a "$REL/scripts/osr_static/run_benchmark.py" "$T/scripts/osr_static/"
cp -a "$REL/scripts/osr_dynamic/run_benchmark.py" "$T/scripts/osr_dynamic/"

for f in README.md launch.sh LICENSE CITATION.bib .gitignore; do
  cp -a "$REL/$f" "$T/$f"
done
cp -a "$REL/docs/BENCHMARK_REPORT.md" "$T/docs/"
cp -a "$0" "$T/scripts/build_release.sh"

cp -a "$EXP/results/STREAM_BASELINES.md" "$T/results/osr_static/" 2>/dev/null || \
  cp -a "$REL/results/osr_static/STREAM_BASELINES.md" "$T/results/osr_static/" 2>/dev/null || true
cp -a "$EXP/results/stream_synthetic/SYNTHETIC_STREAM_TABLE.md" "$T/results/osr_dynamic/" 2>/dev/null || \
  cp -a "$REL/results/osr_dynamic/SYNTHETIC_STREAM_TABLE.md" "$T/results/osr_dynamic/" 2>/dev/null || true

rsync -a --delete "$STAGE/tree/" "$REL/"

chmod +x "$REL/launch.sh" "$REL/scripts/build_release.sh" \
  "$REL/scripts/osr_static/run_benchmark.py" \
  "$REL/scripts/osr_dynamic/run_benchmark.py"

(
  cd "$REL"
  find . -type f ! -path './MANIFEST.sha256' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
)

du -sh "$REL"
echo "Release ready at $REL"
