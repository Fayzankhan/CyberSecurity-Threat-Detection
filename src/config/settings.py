import os
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# Environment settings
ENV = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENV == "production"

# API Settings
API_HOST = os.getenv("API_HOST", "http://127.0.0.1:8000")
PORT = int(os.getenv("PORT", 8000))

# Path settings
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Model paths
MODEL_BINARY_PATH = ARTIFACTS_DIR / "model.joblib"
MODEL_MULTICLASS_PATH = ARTIFACTS_DIR / "model_multiclass.joblib"
METRICS_BINARY_PATH = ARTIFACTS_DIR / "metrics.json"
METRICS_MULTICLASS_PATH = ARTIFACTS_DIR / "metrics_multiclass.json"

# --- Cloud / production security (see .env.example) ---
# If set, clients must send Authorization: Bearer <token> or X-API-Key: <token>
_raw_key = os.getenv("API_SECRET_KEY") or os.getenv("API_KEY")
API_SECRET_KEY = _raw_key.strip() if _raw_key and _raw_key.strip() else None

# Comma-separated origins. Use explicit URLs in production (e.g. https://your-app.streamlit.app).
CORS_ORIGINS = _split_csv(os.getenv("CORS_ORIGINS", "*"))
if not CORS_ORIGINS:
    CORS_ORIGINS = ["*"]

# In-memory rate limit per client IP (HTTP). 0 = disabled.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))

# Behind Render/nginx: use X-Forwarded-For for client IP (only if you trust the proxy).
TRUST_PROXY_HEADERS = _env_bool("TRUST_PROXY_HEADERS", default=IS_PRODUCTION)

# Hide Swagger/OpenAPI in production unless explicitly enabled
EXPOSE_OPENAPI = _env_bool("EXPOSE_OPENAPI", default=not IS_PRODUCTION)

# Legacy name kept for imports that expect it
ALLOWED_HOSTS = ["*"]
