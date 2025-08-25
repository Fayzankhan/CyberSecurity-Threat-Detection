import os
from pathlib import Path
import joblib

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"

def ensure_model_loaded(model_path):
    """Ensures model is available, downloads if not present"""
    if not model_path.exists():
        # For cloud deployment, we'll include models in the repo
        raise RuntimeError(
            f"Model not found at {model_path}. "
            "Please run: python src/train.py"
        )
    return joblib.load(model_path)

# Global model instances
_binary_model = None
_multiclass_model = None

def get_binary_model():
    global _binary_model
    if _binary_model is None:
        model_path = ARTIFACTS / "model.joblib"
        _binary_model = ensure_model_loaded(model_path)
    return _binary_model

def get_multiclass_model():
    global _multiclass_model
    if _multiclass_model is None:
        model_path = ARTIFACTS / "model_multiclass.joblib"
        _multiclass_model = ensure_model_loaded(model_path)
    return _multiclass_model
