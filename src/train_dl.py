"""
Train PyTorch 1D-CNN or LSTM classifiers on NSL-KDD (same 41 raw features as RandomForest).

Preprocessed vector length L = one-hot categoricals + scaled numerics; models treat it as a sequence of length L.

Produces artifacts:
  artifacts/preprocess_dl_binary.joblib
  artifacts/model_dl_binary.pt
  artifacts/metrics_dl_binary.json
  (same names for multiclass)

Usage:
  python -m src.train_dl                  # default: 1D CNN
  python -m src.train_dl --arch lstm
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from .models.dl_tabular import TabularCNN1d, TabularLSTM
from .train import (
    ARTIFACTS,
    add_coarse_attack,
    binarize_label,
    download_if_needed,
    load_nsl_kdd,
)
from .utils.columns import ALL_FEATURES, CATEGORICAL, COARSE_CLASSES, NUMERIC

DROPOUT = 0.2
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS_BINARY = 40
EPOCHS_MULTI = 45
# Short runs for Render/CI (valid checkpoints; lower quality than full training)
EPOCHS_BINARY_QUICK = 3
EPOCHS_MULTI_QUICK = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ],
        verbose_feature_names_out=False,
    )


def _to_float32_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _make_binary_model(arch: str, in_dim: int) -> nn.Module:
    if arch == "cnn1d":
        return TabularCNN1d(in_dim, n_classes=1, dropout=DROPOUT)
    if arch == "lstm":
        return TabularLSTM(
            in_dim,
            n_classes=1,
            hidden=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            dropout=DROPOUT,
        )
    raise ValueError(f"Unknown arch: {arch}")


def _make_multiclass_model(arch: str, in_dim: int, n_classes: int) -> nn.Module:
    if arch == "cnn1d":
        return TabularCNN1d(in_dim, n_classes=n_classes, dropout=DROPOUT)
    if arch == "lstm":
        return TabularLSTM(
            in_dim,
            n_classes=n_classes,
            hidden=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            dropout=DROPOUT,
        )
    raise ValueError(f"Unknown arch: {arch}")


def _binary_ckpt_extra(arch: str) -> dict:
    d: dict = {"architecture": arch, "dropout": DROPOUT}
    if arch == "lstm":
        d["lstm_hidden"] = LSTM_HIDDEN
        d["lstm_layers"] = LSTM_LAYERS
    return d


def train_binary(arch: str, *, epochs: int | None = None) -> None:
    n_epochs = EPOCHS_BINARY if epochs is None else epochs
    download_if_needed()
    train_df, test_df = load_nsl_kdd()
    train_df = binarize_label(train_df)
    test_df = binarize_label(test_df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["target"].values.astype(np.float32)
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["target"].values.astype(np.float32)

    pre = build_preprocessor()
    Xtr = _to_float32_dense(pre.fit_transform(X_train))
    Xte = _to_float32_dense(pre.transform(X_test))
    in_dim = Xtr.shape[1]

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)

    Xt = torch.from_numpy(Xtr).to(DEVICE)
    yt = torch.from_numpy(y_train).to(DEVICE)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    model = _make_binary_model(arch, in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    log_every = max(1, n_epochs // 4)
    for epoch in range(n_epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"  binary [{arch}] epoch {epoch + 1}/{n_epochs}  loss={total / len(Xtr):.4f}")

    model.eval()
    with torch.no_grad():
        Xte_t = torch.from_numpy(Xte).to(DEVICE)
        logits = model(Xte_t).cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        y_pred = (prob >= 0.5).astype(int)
    roc = float(roc_auc_score(y_test, prob))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, ARTIFACTS / "preprocess_dl_binary.joblib")
    payload = {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "task": "binary",
        **_binary_ckpt_extra(arch),
    }
    torch.save(payload, ARTIFACTS / "model_dl_binary.pt")
    with open(ARTIFACTS / "metrics_dl_binary.json", "w") as f:
        json.dump(
            {"roc_auc": roc, "classification_report": report, "confusion_matrix": cm, "architecture": arch},
            f,
            indent=2,
        )

    print("=== Deep learning (binary) complete ===")
    print(f"Architecture: {arch}")
    print(f"Preprocessor: {ARTIFACTS / 'preprocess_dl_binary.joblib'}")
    print(f"Model: {ARTIFACTS / 'model_dl_binary.pt'}")
    print(f"ROC-AUC: {roc:.4f}")


def train_multiclass(arch: str, *, epochs: int | None = None) -> None:
    n_epochs = EPOCHS_MULTI if epochs is None else epochs
    download_if_needed()
    train_df, test_df = load_nsl_kdd()
    train_df = add_coarse_attack(train_df)
    test_df = add_coarse_attack(test_df)

    class_to_idx = {c: i for i, c in enumerate(COARSE_CLASSES)}

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["attack_type"].map(lambda x: class_to_idx.get(str(x).strip(), 0)).values.astype(np.int64)
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["attack_type"].map(lambda x: class_to_idx.get(str(x).strip(), 0)).values.astype(np.int64)

    pre = build_preprocessor()
    Xtr = _to_float32_dense(pre.fit_transform(X_train))
    Xte = _to_float32_dense(pre.transform(X_test))
    in_dim = Xtr.shape[1]
    n_classes = len(COARSE_CLASSES)

    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    class_w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    Xt = torch.from_numpy(Xtr).to(DEVICE)
    yt = torch.from_numpy(y_train).to(DEVICE)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    model = _make_multiclass_model(arch, in_dim, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=class_w)

    model.train()
    log_every = max(1, n_epochs // 3)
    for epoch in range(n_epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"  multiclass [{arch}] epoch {epoch + 1}/{n_epochs}  loss={total / len(Xtr):.4f}")

    model.eval()
    with torch.no_grad():
        Xte_t = torch.from_numpy(Xte).to(DEVICE)
        logits = model(Xte_t).cpu().numpy()
        proba = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba /= proba.sum(axis=1, keepdims=True)
        y_pred_idx = proba.argmax(axis=1)
    y_pred_labels = [COARSE_CLASSES[i] for i in y_pred_idx]
    y_test_labels = [COARSE_CLASSES[i] for i in y_test]

    report = classification_report(y_test_labels, y_pred_labels, output_dict=True, labels=COARSE_CLASSES, zero_division=0)
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=COARSE_CLASSES).tolist()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, ARTIFACTS / "preprocess_dl_multiclass.joblib")
    payload = {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "n_classes": n_classes,
        "classes": COARSE_CLASSES,
        "task": "multiclass",
        **_binary_ckpt_extra(arch),
    }
    torch.save(payload, ARTIFACTS / "model_dl_multiclass.pt")
    with open(ARTIFACTS / "metrics_dl_multiclass.json", "w") as f:
        json.dump(
            {"classes": COARSE_CLASSES, "classification_report": report, "confusion_matrix": cm, "architecture": arch},
            f,
            indent=2,
        )

    print("=== Deep learning (multiclass) complete ===")
    print(f"Architecture: {arch}")
    print(f"Preprocessor: {ARTIFACTS / 'preprocess_dl_multiclass.joblib'}")
    print(f"Model: {ARTIFACTS / 'model_dl_multiclass.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D-CNN or LSTM on NSL-KDD (preprocessed tabular).")
    parser.add_argument(
        "--arch",
        choices=["cnn1d", "lstm"],
        default="cnn1d",
        help="cnn1d: Conv1d stack over feature dimension; lstm: sequence of L steps with 1 input dim",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Few epochs for deploy/CI ({EPOCHS_BINARY_QUICK}/{EPOCHS_MULTI_QUICK}); full quality: omit this flag.",
    )
    args = parser.parse_args()
    if args.quick:
        print("=== train_dl --quick: shorter training (deploy-friendly) ===")
        train_binary(args.arch, epochs=EPOCHS_BINARY_QUICK)
        train_multiclass(args.arch, epochs=EPOCHS_MULTI_QUICK)
    else:
        train_binary(args.arch)
        train_multiclass(args.arch)


if __name__ == "__main__":
    main()
