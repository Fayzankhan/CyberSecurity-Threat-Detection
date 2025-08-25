import os

# API configuration
API_HOST = os.getenv('API_HOST', 'http://127.0.0.1:8000')

# Model paths
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
MODEL_BINARY_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')
MODEL_MULTICLASS_PATH = os.path.join(ARTIFACTS_DIR, 'model_multiclass.joblib')
METRICS_BINARY_PATH = os.path.join(ARTIFACTS_DIR, 'metrics.json')
METRICS_MULTICLASS_PATH = os.path.join(ARTIFACTS_DIR, 'metrics_multiclass.json')
