"""Inference for PyTorch 1D-CNN / LSTM tabular models (same feature schema as sklearn)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from .models.dl_tabular import load_tabular_binary, load_tabular_multiclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def predict_binary_dl(
    df: pd.DataFrame,
    feature_cols: List[str],
    preprocess_path: Path,
    model_path: Path,
) -> Tuple[List[int], List[float]]:
    pre = joblib.load(preprocess_path)
    model, _ = load_tabular_binary(model_path, device=DEVICE)
    X = _dense(pre.transform(df[feature_cols]))
    xt = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        logits = model(xt).cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int).tolist()
    return pred, prob.tolist()


def predict_multiclass_dl(
    df: pd.DataFrame,
    feature_cols: List[str],
    preprocess_path: Path,
    model_path: Path,
) -> Tuple[List[str], List[float]]:
    pre = joblib.load(preprocess_path)
    model, ckpt = load_tabular_multiclass(model_path, device=DEVICE)
    classes: List[str] = ckpt["classes"]
    X = _dense(pre.transform(df[feature_cols]))
    xt = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        logits = model(xt).cpu().numpy()
    proba = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba /= proba.sum(axis=1, keepdims=True)
    conf = proba.max(axis=1).tolist()
    idx = proba.argmax(axis=1)
    preds = [classes[i] for i in idx]
    return preds, conf
