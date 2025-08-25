from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd

from ..utils.columns import ALL_FEATURES
from ..config.settings import ARTIFACTS_DIR, IS_PRODUCTION, PORT, ALLOWED_HOSTS

ARTIFACTS = ARTIFACTS_DIR

class Event(BaseModel):
    # Define all 41 features
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: float
    dst_bytes: float
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: float
    srv_count: float
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: float
    dst_host_srv_count: float
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

app = FastAPI(title="Cyber Threat Detector", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from .model_loader import get_binary_model as get_model

@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        # Try to load models to verify they're accessible
        get_model()
        get_model_multi()
        return {"status": "ok", "message": "API is healthy and models are loaded"}
    except Exception as e:
        return {
            "status": "warning",
            "message": f"API is running but encountered an issue: {str(e)}"
        }

@app.post("/predict")
def predict(evt: Event):
    """Single prediction endpoint"""
    try:
        model = get_model()
        X = [[getattr(evt, f) for f in ALL_FEATURES]]
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        return {
            "attack_probability": prob,
            "is_attack": pred
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

from .model_loader import get_multiclass_model as get_model_multi

class BatchEvents(BaseModel):
    records: list[dict]

@app.post("/predict-batch")
def predict_batch(payload: BatchEvents):
    """Batch prediction endpoint"""
    try:
        model = get_model()
        df = pd.DataFrame(payload.records)
        
        # Validate input features
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )
        
        # Make predictions
        prob = model.predict_proba(df[ALL_FEATURES])[:, 1]
        pred = (prob >= 0.5).astype(int).tolist()
        
        return {
            "predictions": pred,
            "probabilities": prob.tolist()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/predict-multiclass")
def predict_multiclass(payload: BatchEvents):
    """Multiclass prediction endpoint"""
    try:
        model = get_model_multi()
        df = pd.DataFrame(payload.records)
        
        # Validate input features
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )
        
        # Make predictions
        pred = model.predict(df[ALL_FEATURES]).tolist()
        
        # Get confidence scores if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df[ALL_FEATURES])
            proba_list = np.max(proba, axis=1).tolist()
        else:
            proba_list = [None] * len(pred)
        
        return {
            "predictions": pred,
            "confidence": proba_list
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Multiclass prediction failed: {str(e)}"
        )