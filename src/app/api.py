from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Literal
import json
import logging
import os
import pandas as pd
from collections import Counter

from ..config.settings import (
    API_SECRET_KEY,
    ARTIFACTS_DIR,
    CORS_ORIGINS,
    EXPOSE_OPENAPI,
    RATE_LIMIT_PER_MINUTE,
    ROOT_DIR,
)

from .predict_core import (
    BINARY_MODEL_PATH,
    MULTICLASS_MODEL_PATH,
    DL_BINARY_PRE_PATH,
    DL_BINARY_MODEL_PATH,
    DL_MULTICLASS_PRE_PATH,
    DL_MULTICLASS_MODEL_PATH,
    load_binary_model,
    load_multiclass_model,
    run_binary_predict,
    run_multiclass_predict,
)
from .render_dl_bootstrap import bootstrap_thread_alive, maybe_bootstrap_dl_async
from .security_middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    websocket_api_key_authorized,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_docs = "/docs" if EXPOSE_OPENAPI else None
_redoc = "/redoc" if EXPOSE_OPENAPI else None
_openapi = "/openapi.json" if EXPOSE_OPENAPI else None

app = FastAPI(
    title="Cyber Threat Detector",
    version="1.0.0",
    docs_url=_docs,
    redoc_url=_redoc,
    openapi_url=_openapi,
)

# Innermost first: API key -> rate limit -> CORS -> security headers (outermost)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)


class BatchEvents(BaseModel):
    records: List[Dict[str, Any]]
    model_backend: Literal["sklearn", "deep"] = "sklearn"


@app.on_event("startup")
async def startup_event():
    """Preload models on startup for faster first prediction"""
    logger.info("Starting up API...")
    if API_SECRET_KEY:
        logger.info("API key authentication is enabled (set API_SECRET_KEY).")
    if RATE_LIMIT_PER_MINUTE > 0:
        logger.info("Rate limit: %s requests/minute per IP (HTTP)", RATE_LIMIT_PER_MINUTE)
    try:
        if BINARY_MODEL_PATH.exists():
            logger.info("Preloading binary model...")
            load_binary_model()
        else:
            logger.warning(f"Binary model not found at {BINARY_MODEL_PATH}")

        if MULTICLASS_MODEL_PATH.exists():
            logger.info("Preloading multiclass model...")
            load_multiclass_model()
        else:
            logger.warning(f"Multiclass model not found at {MULTICLASS_MODEL_PATH}")
        if DL_BINARY_MODEL_PATH.exists():
            logger.info("Deep learning binary artifacts present (loaded on first deep request).")
        if DL_MULTICLASS_MODEL_PATH.exists():
            logger.info("Deep learning multiclass artifacts present (loaded on first deep request).")
        maybe_bootstrap_dl_async(ROOT_DIR, DL_BINARY_MODEL_PATH, DL_MULTICLASS_MODEL_PATH)
        if ARTIFACTS_DIR.is_dir():
            logger.info("artifacts/: %s", sorted(p.name for p in ARTIFACTS_DIR.iterdir()))
        logger.info("API startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.exception(e)


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        binary_available = BINARY_MODEL_PATH.exists()
        multiclass_available = MULTICLASS_MODEL_PATH.exists()

        dl_bin = DL_BINARY_PRE_PATH.exists() and DL_BINARY_MODEL_PATH.exists()
        dl_multi = DL_MULTICLASS_PRE_PATH.exists() and DL_MULTICLASS_MODEL_PATH.exists()
        booting = bootstrap_thread_alive()

        def _deep_status(ready: bool) -> str:
            if ready:
                return "available"
            if booting:
                return "initializing"
            return "not found"

        status = {
            "status": "ok",
            "message": "API is running",
            "security": {
                "api_key_required": bool(API_SECRET_KEY),
                "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE if RATE_LIMIT_PER_MINUTE > 0 else None,
            },
            "streaming": {
                "websocket_predict": "/ws/predict",
                "description": "Micro-batch IoT-style events over WebSocket (JSON per message).",
            },
            "models": {
                "binary": "available" if binary_available else "not found",
                "multiclass": "available" if multiclass_available else "not found",
                "binary_deep": _deep_status(dl_bin),
                "multiclass_deep": _deep_status(dl_multi),
            },
        }
        if os.environ.get("RENDER"):
            status["models"]["deep_bootstrap_note"] = (
                "If binary_deep stays 'initializing', training is still running (~2–8 min on CPU). "
                "If it stays 'not found' with no initializing, check Render logs for train_dl errors."
            )
        return status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/predict-batch")
async def predict_batch(payload: BatchEvents):
    """Binary classification batch prediction endpoint"""
    try:
        logger.info("Received binary prediction request")
        logger.info(f"Received {len(payload.records)} records")

        try:
            df = pd.DataFrame(payload.records)
            logger.info(f"Created DataFrame with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid data format: {str(e)}"},
            )

        try:
            return run_binary_predict(df, payload.model_backend)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except FileNotFoundError as e:
            return JSONResponse(status_code=404, content={"error": str(e)})
        except Exception as e:
            logger.error(f"Binary prediction failed: {str(e)}")
            logger.exception(e)
            return JSONResponse(
                status_code=500,
                content={"error": f"Binary prediction failed: {str(e)}"},
            )

    except Exception as e:
        logger.error(f"Unexpected error in binary prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"},
        )


@app.post("/predict-multiclass")
async def predict_multiclass(payload: BatchEvents):
    """Multiclass prediction endpoint"""
    try:
        logger.info("Received multiclass prediction request")
        logger.info(f"Received {len(payload.records)} records")

        try:
            df = pd.DataFrame(payload.records)
            logger.info(f"Created DataFrame with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid data format: {str(e)}"},
            )

        try:
            out = run_multiclass_predict(df, payload.model_backend)
            pred_counts = Counter(out["predictions"])
            logger.info(f"Predictions summary: {dict(pred_counts)}")
            return out
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except FileNotFoundError as e:
            return JSONResponse(status_code=404, content={"error": str(e)})
        except Exception as e:
            logger.error(f"Multiclass prediction failed: {str(e)}")
            logger.exception(e)
            return JSONResponse(
                status_code=500,
                content={"error": f"Multiclass prediction failed: {str(e)}"},
            )

    except Exception as e:
        logger.error(f"Unexpected error in multiclass prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"},
        )


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    Real-time micro-batch predictions for IoT-style streams.

    Each text message is JSON:
      {"task": "binary"|"multiclass", "model_backend": "sklearn"|"deep", "records": [ {...}, ... ]}

    When API_SECRET_KEY is set, send the same token as HTTP: Authorization Bearer, X-API-Key, or ?api_key=
    """
    if API_SECRET_KEY and not websocket_api_key_authorized(websocket):
        await websocket.close(code=1008)
        logger.warning("WebSocket rejected: bad API key")
        return

    await websocket.accept()
    logger.info("WebSocket /ws/predict connected")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                await websocket.send_json({"ok": False, "error": f"Invalid JSON: {e}"})
                continue

            task = msg.get("task", "binary")
            backend = msg.get("model_backend", "sklearn")
            records = msg.get("records")
            if not isinstance(records, list) or len(records) == 0:
                await websocket.send_json({"ok": False, "error": "Expected non-empty 'records' array"})
                continue

            try:
                df = pd.DataFrame(records)
            except Exception as e:
                await websocket.send_json({"ok": False, "error": f"Invalid records: {e}"})
                continue

            try:
                if task == "binary":
                    out = run_binary_predict(df, backend)
                elif task == "multiclass":
                    out = run_multiclass_predict(df, backend)
                else:
                    await websocket.send_json(
                        {"ok": False, "error": "task must be 'binary' or 'multiclass'"}
                    )
                    continue
                await websocket.send_json({"ok": True, **out})
            except ValueError as e:
                await websocket.send_json({"ok": False, "error": str(e)})
            except FileNotFoundError as e:
                await websocket.send_json({"ok": False, "error": str(e)})
            except Exception as e:
                logger.exception("WebSocket prediction error")
                await websocket.send_json({"ok": False, "error": str(e)})
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/predict disconnected")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
