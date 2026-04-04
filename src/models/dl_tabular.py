"""1D-CNN and LSTM models for preprocessed tabular NSL-KDD vectors (length L after one-hot + scaling)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularCNN1d(nn.Module):
    """Treats the feature vector as a 1D signal: (B, L) -> Conv1d over length L."""

    def __init__(self, seq_len: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.n_out = n_classes
        _ = seq_len  # fixed at train time; kept for API symmetry
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        out_dim = 1 if n_classes <= 1 else n_classes
        self.head = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        h = self.encoder(x).squeeze(-1)
        logits = self.head(h)
        if self.n_out <= 1:
            return logits.squeeze(-1)
        return logits


class TabularLSTM(nn.Module):
    """Each preprocessed dimension is one timestep with a single input channel: (B, L) -> (B, L, 1)."""

    def __init__(
        self,
        seq_len: int,
        n_classes: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        _ = seq_len
        self.n_out = n_classes
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = 1 if n_classes <= 1 else n_classes
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        logits = self.head(h)
        if self.n_out <= 1:
            return logits.squeeze(-1)
        return logits


def load_tabular_binary(path: str | object, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    arch = ckpt["architecture"]
    in_dim = ckpt["in_dim"]
    drop = ckpt.get("dropout", 0.2)
    if arch == "cnn1d":
        model = TabularCNN1d(in_dim, n_classes=1, dropout=drop)
    elif arch == "lstm":
        model = TabularLSTM(
            in_dim,
            n_classes=1,
            hidden=ckpt.get("lstm_hidden", 128),
            num_layers=ckpt.get("lstm_layers", 2),
            dropout=drop,
        )
    else:
        raise ValueError(f"Unknown architecture in checkpoint: {arch}")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def load_tabular_multiclass(path: str | object, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    arch = ckpt["architecture"]
    in_dim = ckpt["in_dim"]
    n_classes = ckpt["n_classes"]
    drop = ckpt.get("dropout", 0.2)
    if arch == "cnn1d":
        model = TabularCNN1d(in_dim, n_classes=n_classes, dropout=drop)
    elif arch == "lstm":
        model = TabularLSTM(
            in_dim,
            n_classes=n_classes,
            hidden=ckpt.get("lstm_hidden", 128),
            num_layers=ckpt.get("lstm_layers", 2),
            dropout=drop,
        )
    else:
        raise ValueError(f"Unknown architecture in checkpoint: {arch}")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
