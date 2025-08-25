from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd

from ..utils.columns import ALL_FEATURES

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"

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

_model = None

def get_model():
    global _model
    if _model is None:
        model_path = ARTIFACTS / "model.joblib"
        if not model_path.exists():
            raise RuntimeError("Model not found. Train it with: python src/train.py")
        _model = joblib.load(model_path)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(evt: Event):
    model = get_model()
    X = [[getattr(evt, f) for f in ALL_FEATURES]]
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    return {
        "attack_probability": prob,
        "is_attack": pred
    }


_model_multi = None

def get_model_multi():
    global _model_multi
    if _model_multi is None:
        model_path = ARTIFACTS / "model_multiclass.joblib"
        if not model_path.exists():
            raise RuntimeError("Multiclass model not found. Train it with: python src/train.py")
        _model_multi = joblib.load(model_path)
    return _model_multi


class BatchEvents(BaseModel):
    records: list[dict]

@app.post("/predict-batch")
def predict_batch(payload: BatchEvents):
    model = get_model()
    try:
        df = pd.DataFrame(payload.records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid records: {e}")
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    prob = model.predict_proba(df[ALL_FEATURES])[:, 1]
    pred = (prob >= 0.5).astype(int).tolist()
    return {"predictions": pred, "probabilities": prob.tolist()}

@app.post("/predict-multiclass")
def predict_multiclass(payload: BatchEvents):
    model = get_model_multi()
    try:
        df = pd.DataFrame(payload.records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid records: {e}")
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    pred = model.predict(df[ALL_FEATURES]).tolist()
    # Try to get class probabilities if supported
    if hasattr(model, "predict_proba"):
        import numpy as np
        proba = model.predict_proba(df[ALL_FEATURES])
        proba_list = np.max(proba, axis=1).tolist()
    else:
        proba_list = [None] * len(pred)
    return {"predictions": pred, "confidence": proba_list}
