#!/usr/bin/env python3
"""Verify DL files exist using the same paths as FastAPI (avoids shell cwd / ROOT mismatch)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.app.predict_core import (  # noqa: E402
    DL_BINARY_MODEL_PATH,
    DL_BINARY_PRE_PATH,
    DL_MULTICLASS_MODEL_PATH,
    DL_MULTICLASS_PRE_PATH,
)


def main() -> int:
    paths = [
        DL_BINARY_PRE_PATH,
        DL_BINARY_MODEL_PATH,
        DL_MULTICLASS_PRE_PATH,
        DL_MULTICLASS_MODEL_PATH,
    ]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print("Missing DL artifacts (API will report binary_deep: not found):", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        parent = DL_BINARY_MODEL_PATH.parent
        print(f"\nContents of {parent}:", file=sys.stderr)
        if parent.is_dir():
            for p in sorted(parent.iterdir()):
                print(f"  {p.name}", file=sys.stderr)
        else:
            print("  (directory does not exist)", file=sys.stderr)
        return 1
    for p in paths:
        print(f"OK {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
