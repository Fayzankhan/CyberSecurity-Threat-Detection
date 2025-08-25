import argparse
import joblib
import pandas as pd
from pathlib import Path

from .utils.columns import ALL_FEATURES

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"

def main():
    parser = argparse.ArgumentParser(description="Batch predict cyber threats from CSV logs.")
    parser.add_argument("--input", required=True, help="Path to input CSV with 41 feature columns.")
    parser.add_argument("--output", required=True, help="Path to save output CSV with predictions.")
    args = parser.parse_args()

    model_path = ARTIFACTS / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run: python src/train.py")

    pipe = joblib.load(model_path)

    df = pd.read_csv(args.input)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}\nExpected columns: {ALL_FEATURES}")

    proba = pipe.predict_proba(df[ALL_FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["attack_probability"] = proba
    out["is_attack"] = pred

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
