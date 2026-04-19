"""Shared batch prediction logic used by HTTP and WebSocket endpoints."""

from __future__ import annotations

from typing import Any, Dict

import joblib
import pandas as pd

from ..utils.columns import ALL_FEATURES
from ..config.settings import ARTIFACTS_DIR

ARTIFACTS = ARTIFACTS_DIR

BINARY_MODEL_PATH = ARTIFACTS / "model.joblib"
MULTICLASS_MODEL_PATH = ARTIFACTS / "model_multiclass.joblib"
DL_BINARY_PRE_PATH = ARTIFACTS / "preprocess_dl_binary.joblib"
DL_BINARY_MODEL_PATH = ARTIFACTS / "model_dl_binary.pt"
DL_MULTICLASS_PRE_PATH = ARTIFACTS / "preprocess_dl_multiclass.joblib"
DL_MULTICLASS_MODEL_PATH = ARTIFACTS / "model_dl_multiclass.pt"

_binary_model = None
_multiclass_model = None


def load_binary_model():
    global _binary_model
    if _binary_model is None:
        if not BINARY_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Binary model not found at {BINARY_MODEL_PATH}. "
                "Please train the model first by running: python -m src.train"
            )
        _binary_model = joblib.load(BINARY_MODEL_PATH)
    return _binary_model


def load_multiclass_model():
    global _multiclass_model
    if _multiclass_model is None:
        if not MULTICLASS_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Multiclass model not found at {MULTICLASS_MODEL_PATH}. "
                "Please train the model first by running: python -m src.train"
            )
        _multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
    return _multiclass_model


def validate_feature_frame(df: pd.DataFrame) -> str | None:
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}"
    return None


def run_binary_predict(df: pd.DataFrame, backend: str) -> Dict[str, Any]:
    if backend not in ("sklearn", "deep"):
        raise ValueError("model_backend must be 'sklearn' or 'deep'")
    err = validate_feature_frame(df)
    if err:
        raise ValueError(err)
    if backend == "deep":
        if DL_BINARY_PRE_PATH.exists() and DL_BINARY_MODEL_PATH.exists():
            from ..dl_backend import predict_binary_dl

            predictions, probabilities = predict_binary_dl(
                df, ALL_FEATURES, DL_BINARY_PRE_PATH, DL_BINARY_MODEL_PATH
            )
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "model_backend": "deep",
            }
        # Fast production-safe fallback: keep predictions working even when DL artifacts are absent.
        backend = "sklearn"

    model = load_binary_model()
    X = df[ALL_FEATURES]
    predictions = model.predict(X).tolist()
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1].tolist()
    else:
        probabilities = predictions
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "model_backend": "sklearn",
    }


def run_multiclass_predict(df: pd.DataFrame, backend: str) -> Dict[str, Any]:
    if backend not in ("sklearn", "deep"):
        raise ValueError("model_backend must be 'sklearn' or 'deep'")
    err = validate_feature_frame(df)
    if err:
        raise ValueError(err)
    if backend == "deep":
        if DL_MULTICLASS_PRE_PATH.exists() and DL_MULTICLASS_MODEL_PATH.exists():
            from ..dl_backend import predict_multiclass_dl

            predictions, confidence = predict_multiclass_dl(
                df, ALL_FEATURES, DL_MULTICLASS_PRE_PATH, DL_MULTICLASS_MODEL_PATH
            )
            return {
                "predictions": predictions,
                "confidence": confidence,
                "model_backend": "deep",
            }
        # Fast production-safe fallback: keep predictions working even when DL artifacts are absent.
        backend = "sklearn"

    model = load_multiclass_model()
    X = df[ALL_FEATURES]
    predictions = model.predict(X).tolist()
    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(X)
        confidence = proba_matrix.max(axis=1).tolist()
    else:
        confidence = [1.0] * len(predictions)
    return {
        "predictions": predictions,
        "confidence": confidence,
        "model_backend": "sklearn",
    }
