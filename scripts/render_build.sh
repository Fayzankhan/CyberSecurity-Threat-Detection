#!/usr/bin/env bash
# Used by Render (see render.yaml). Fails the build if DL artifacts are missing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

echo "=== train_dl --quick (writes PyTorch backend artifacts) ==="
python -m src.train_dl --quick

echo "=== verify artifacts ==="
test -f artifacts/model_dl_binary.pt
test -f artifacts/preprocess_dl_binary.joblib
test -f artifacts/model_dl_multiclass.pt
test -f artifacts/preprocess_dl_multiclass.joblib
ls -la artifacts/
echo "=== Render build OK ==="
