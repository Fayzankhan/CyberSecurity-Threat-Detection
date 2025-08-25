import time
import json
import threading
import argparse
from pathlib import Path

import pandas as pd
import requests

from ..utils.columns import ALL_FEATURES, CSV_COLUMNS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

def ensure_dataset():
    train = DATA_DIR / "KDDTrain+.txt"
    test  = DATA_DIR / "KDDTest+.txt"
    if not train.exists() or not test.exists():
        print("Dataset not found. Run: python src/train.py (it will download data).")
        raise SystemExit(1)
    return test

def stream(file_path: Path, rate: float, url: str):
    df = pd.read_csv(file_path, names=CSV_COLUMNS)
    # Build events dicts
    events = df[ALL_FEATURES].to_dict(orient="records")
    delay = 1.0 / rate if rate > 0 else 0
    print(f"Streaming {len(events)} events to {url} at ~{rate} eps...")
    for i, evt in enumerate(events):
        try:
            r = requests.post(url, json=evt, timeout=2)
            if r.ok:
                res = r.json()
                alert = "ALERT 🚨" if res.get("is_attack") else "ok"
                print(f"[{i:05d}] {alert}  p={res.get('attack_probability'):.3f}")
            else:
                print(f"[{i:05d}] HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"[{i:05d}] error: {e}")
        time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description="Simulate a live event stream to the FastAPI /predict endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict", help="Prediction endpoint URL.")
    parser.add_argument("--rate", type=float, default=4.0, help="Events per second.")
    args = parser.parse_args()

    test_path = ensure_dataset()
    stream(test_path, rate=args.rate, url=args.url)

if __name__ == "__main__":
    main()
