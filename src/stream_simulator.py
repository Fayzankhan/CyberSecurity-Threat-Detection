"""
Push NSL-KDD rows to the API WebSocket as micro-batches (simulates IoT / edge event streams).

Requires: API running with `uvicorn src.app.api:app`, and trained sklearn models.

  python -m src.stream_simulator --url ws://127.0.0.1:8000/ws/predict --batch-size 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets", file=sys.stderr)
    raise


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ws_extra_headers(api_key: str | None) -> list[tuple[str, str]]:
    key = api_key or os.environ.get("API_SECRET_KEY") or os.environ.get("API_KEY")
    if not key:
        return []
    return [("Authorization", f"Bearer {key.strip()}")]


async def run(
    url: str,
    batch_size: int,
    max_rows: int | None,
    task: str,
    backend: str,
    api_key: str | None,
) -> None:
    root = _project_root()
    data_path = root / "data" / "KDDTest+.txt"
    if not data_path.exists():
        print(f"Dataset not found: {data_path}. Run download or python -m src.train first.", file=sys.stderr)
        sys.exit(1)

    from src.utils.columns import ALL_FEATURES, CSV_COLUMNS

    df = pd.read_csv(data_path, names=CSV_COLUMNS, nrows=max_rows)
    n = len(df)
    print(f"Streaming {n} events in batches of {batch_size} to {url} ({task}, {backend})")

    extra = _ws_extra_headers(api_key)
    connect_kw = {"additional_headers": extra} if extra else {}
    async with websockets.connect(url, **connect_kw) as ws:
        for start in range(0, n, batch_size):
            chunk = df.iloc[start : start + batch_size]
            payload = {
                "task": task,
                "model_backend": backend,
                "records": chunk[ALL_FEATURES].to_dict(orient="records"),
            }
            await ws.send(json.dumps(payload))
            raw = await ws.recv()
            data = json.loads(raw)
            if not data.get("ok"):
                print(f"Batch {start}-{start + len(chunk)} error: {data.get('error')}")
            else:
                preds = data.get("predictions", [])
                print(f"Batch {start:5d}-{start + len(chunk):5d}  ok  records={len(preds)}")


def main() -> None:
    p = argparse.ArgumentParser(description="WebSocket micro-batch stream simulator")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/predict", help="WebSocket URL")
    p.add_argument("--batch-size", type=int, default=25, help="Events per message")
    p.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Total rows to send (0 = entire KDDTest+ file)",
    )
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--model-backend", choices=["sklearn", "deep"], default="sklearn")
    p.add_argument(
        "--api-key",
        default=None,
        help="Bearer token (defaults to API_SECRET_KEY / API_KEY env)",
    )
    args = p.parse_args()
    max_rows = None if args.max_rows == 0 else args.max_rows
    asyncio.run(run(args.url, args.batch_size, max_rows, args.task, args.model_backend, args.api_key))


if __name__ == "__main__":
    main()
