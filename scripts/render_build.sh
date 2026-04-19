#!/usr/bin/env bash
# Used by Render (see render.yaml). Fails the build if DL artifacts are missing.
set -euo pipefail
# Resolve repo root from this script (not from cwd — Render cwd can differ).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

echo "=== train_dl --quick (writes + self-verifies PyTorch artifacts) ==="
python -m src.train_dl --quick
echo "=== Render build OK ==="
