import os
from pathlib import Path
import joblib
from fastapi import HTTPException
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"

def create_default_model():
    """Create a simple default model if no trained model is available"""
    print("Creating default model...")
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    # Fit with dummy data to initialize
    X = np.random.rand(100, 41)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    return model

def ensure_model_loaded(model_path):
    """Ensures model is available, creates default if not present"""
    try:
        if model_path.exists():
            return joblib.load(model_path)
        else:
            print(f"Model not found at {model_path}, creating default model")
            model = create_default_model()
            # Save the model
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return create_default_model()

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