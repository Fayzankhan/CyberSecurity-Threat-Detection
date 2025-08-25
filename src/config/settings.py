import os
from pathlib import Path

# Environment settings
ENV = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENV == "production"

# API Settings
API_HOST = os.getenv("API_HOST", "http://127.0.0.1:8000")
PORT = int(os.getenv("PORT", 8000))

# Path settings
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Model paths
MODEL_BINARY_PATH = ARTIFACTS_DIR / "model.joblib"
MODEL_MULTICLASS_PATH = ARTIFACTS_DIR / "model_multiclass.joblib"
METRICS_BINARY_PATH = ARTIFACTS_DIR / "metrics.json"
METRICS_MULTICLASS_PATH = ARTIFACTS_DIR / "metrics_multiclass.json"

# Security settings
ALLOWED_HOSTS = ["*"]  # In production, you might want to restrict this
if IS_PRODUCTION:
    # Add production-specific settings here
    pass
