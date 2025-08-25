import os
import json
import urllib.request
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .utils.columns import CSV_COLUMNS, ALL_FEATURES, CATEGORICAL, LABEL, DIFFICULTY, ATTACK_TYPE_MAP, COARSE_CLASSES

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
TEST_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH  = DATA_DIR / "KDDTest+.txt"


def download_if_needed() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAIN_PATH.exists():
        print(f"Downloading train set → {TRAIN_PATH} ...")
        urllib.request.urlretrieve(TRAIN_URL, TRAIN_PATH)
    else:
        print("Train set already present.")
    if not TEST_PATH.exists():
        print(f"Downloading test set → {TEST_PATH} ...")
        urllib.request.urlretrieve(TEST_URL, TEST_PATH)
    else:
        print("Test set already present.")


def load_nsl_kdd() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # NSL-KDD files are comma-separated with no header; last two columns are label & difficulty
    colnames = CSV_COLUMNS
    train_df = pd.read_csv(TRAIN_PATH, names=colnames)
    test_df  = pd.read_csv(TEST_PATH, names=colnames)
    return train_df, test_df


def binarize_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = (df[LABEL] != "normal").astype(int)  # 1 = attack, 0 = normal
    return df

def add_coarse_attack(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attack_type"] = df[LABEL].map(lambda x: ATTACK_TYPE_MAP.get(str(x).strip(), "normal"))
    return df

    df = df.copy()
    df["target"] = (df[LABEL] != "normal").astype(int)  # 1 = attack, 0 = normal
    return df


def build_pipeline(categorical: list[str]) -> Pipeline:
    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    clf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    pipe = Pipeline(steps=[
        ("preprocess", preproc),
        ("clf", clf),
    ])
    return pipe


def train_and_evaluate() -> None:
    """Train binary classifier."""

    download_if_needed()
    train_df, test_df = load_nsl_kdd()

    # Prepare data
    train_df = binarize_label(train_df)
    test_df  = binarize_label(test_df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["target"].values
    X_test  = test_df[ALL_FEATURES]
    y_test  = test_df["target"].values

    pipe = build_pipeline(CATEGORICAL)
    pipe.fit(X_train, y_train)

    # Eval
    y_pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, y_prob))
    else:
        y_prob = None
        roc = None

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ARTIFACTS / "model.joblib")
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump({
            "roc_auc": roc,
            "classification_report": report,
            "confusion_matrix": cm,
        }, f, indent=2)

    print("=== Training complete ===")
    print(f"Model saved to: {ARTIFACTS / 'model.joblib'}")
    print(f"Metrics saved to: {ARTIFACTS / 'metrics.json'}")
    if roc is not None:
        print(f"ROC-AUC: {roc:.4f}")


if __name__ == "__main__":
    train_and_evaluate()

def train_multiclass() -> None:
    """Train a multiclass classifier for coarse attack categories."""
    download_if_needed()
    train_df, test_df = load_nsl_kdd()
    train_df = add_coarse_attack(train_df)
    test_df = add_coarse_attack(test_df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["attack_type"].values
    X_test  = test_df[ALL_FEATURES]
    y_test  = test_df["attack_type"].values

    pipe = build_pipeline(CATEGORICAL)
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    try:
        import sklearn
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception:
        classification_report = lambda *a, **k: {}
        confusion_matrix = lambda *a, **k: []

    report = classification_report(y_test, y_pred, output_dict=True, labels=COARSE_CLASSES, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=COARSE_CLASSES).tolist()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    import joblib, json as _json
    joblib.dump(pipe, ARTIFACTS / "model_multiclass.joblib")
    with open(ARTIFACTS / "metrics_multiclass.json", "w") as f:
        _json.dump({
            "classes": COARSE_CLASSES,
            "classification_report": report,
            "confusion_matrix": cm,
        }, f, indent=2)

    print("=== Multiclass training complete ===")
    print(f"Model saved to: {ARTIFACTS / 'model_multiclass.joblib'}")
    print(f"Metrics saved to: {ARTIFACTS / 'metrics_multiclass.json'}")


if __name__ == "__main__":
    # Train both models when run as a script
    train_and_evaluate()
    train_multiclass()
