#!/usr/bin/env bash
# Entry points: setup, static/dynamic baselines, aigas-static, aigas-dynamic.
#
#   bash launch.sh setup
#   bash launch.sh static --cpus 8
#   bash launch.sh dynamic
#   bash launch.sh aigas-static   # needs AIGAS_PHASE1_DIR
#   bash launch.sh aigas-dynamic  # needs NeuPy (garage/ or AIGAS_PYTHON)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  sed -n '2,8p' "$0"
}

CPUS="$(nproc)"
CMD="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus) CPUS="$2"; shift 2 ;;
    -h|--help) CMD="help"; shift ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$CMD" == "help" || "$CMD" == "-h" || "$CMD" == "--help" ]]; then
  usage
  exit 0
fi

ensure_uv() {
  export PATH="$HOME/.local/bin:$PATH"
  if command -v uv >/dev/null 2>&1; then return; fi
  python3 -m pip install --user --quiet uv 2>/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

setup_all() {
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    ensure_uv
    uv venv "$ROOT/.venv" --python 3.12 --seed
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r "$ROOT/requirements.lock.txt"
}

PY="$ROOT/.venv/bin/python"

case "$CMD" in
  setup) setup_all ;;
  static)
    setup_all
    "$PY" "$ROOT/scripts/osr_static/run_benchmark.py" --cpus "$CPUS"
    ;;
  dynamic)
    setup_all
    "$PY" "$ROOT/scripts/osr_dynamic/run_benchmark.py"
    ;;
  # legacy aliases
  stream) setup_all; "$PY" "$ROOT/scripts/osr_static/run_benchmark.py" --cpus "$CPUS" ;;
  stream-synthetic) setup_all; "$PY" "$ROOT/scripts/osr_dynamic/run_benchmark.py" ;;
  aigas-static)
    setup_all
    "$PY" "$ROOT/scripts/osr_static/run_aigas_benchmark.py"
    ;;
  aigas-dynamic)
    setup_all
    "$PY" "$ROOT/scripts/osr_dynamic/run_aigas_benchmark.py"
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    usage >&2
    exit 2
    ;;
esac
