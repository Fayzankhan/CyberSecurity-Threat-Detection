#!/usr/bin/env bash
# Used by Render (see render.yaml). Keep this lean for free-tier reliability.
set -euo pipefail
# Resolve repo root from this script (not from cwd — Render cwd can differ).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

echo "=== skipping DL training on Render free tier (prevents OOM/slow deploys) ==="
echo "=== Render build OK ==="
