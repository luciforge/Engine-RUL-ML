"""Temporal Convolutional Network (TCN) for binary failure classification.

Architecture:
    - Dilated 1-D causal convolution stack (dilations 1, 2, 4, 8)
    - Residual connections between blocks
    - Global average pooling over the time dimension
    - Linear classification head (2 classes)

Key design choices:
    - Reuses SlidingWindowDataset from lstm.py.
    - train_tcn / predict_proba_tcn share the same call signature as
      train_lstm / predict_proba_lstm for interchangeable use in train.py.
    - Runs efficiently on CPU; no GPU required for CMAPSS-scale data.
    - Cosine LR schedule + early stopping on validation PR-AUC (same as LSTM).

Receptive field with default settings (kernel=3, dilations=[1,2,4,8]):
    Each block adds 2*(kernel-1)*dilation = {4, 8, 16, 32} → total 60 time steps
    covered, comfortably within the default window of 30 cycles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from models.deep.lstm import SlidingWindowDataset, _EarlyStopping

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _cfg() -> dict:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    """1-D causal (left-padded) dilated convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel, dilation=dilation, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    """Single TCN residual block: two dilated causal convs + ReLU + dropout + skip."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        # 1x1 projection when channel dimensions differ
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.dropout(self.relu(self.norm1(self.conv1(x))))
        out = self.dropout(self.relu(self.norm2(self.conv2(out))))
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """Dilated Temporal Convolutional Network for sequence classification.

    Parameters
    ----------
    input_size  : number of input features (per time step)
    num_channels: hidden channel width for all TCN blocks
    kernel_size : convolution kernel width
    dilations   : list of dilation factors; one _TCNBlock per entry
    dropout     : dropout probability inside each block
    """

    def __init__(
        self,
        input_size: int,
        num_channels: int = 64,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]

        layers: list[nn.Module] = []
        in_ch = input_size
        for d in dilations:
            layers.append(_TCNBlock(in_ch, num_channels, kernel_size, d, dropout))
            in_ch = num_channels

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape from DataLoader: (B, T, C) — permute to (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        out = self.tcn(x)               # (B, num_channels, T)
        out = out.mean(dim=2)           # global average pool → (B, num_channels)
        return self.head(out)           # (B, 2)


# ---------------------------------------------------------------------------
# Training & inference
# ---------------------------------------------------------------------------

def train_tcn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_channels: int = 64,
    kernel_size: int = 3,
    dilations: list[int] | None = None,
    dropout: float = 0.2,
    device: str | None = None,
) -> TCNModel:
    """Train TCNModel with cosine LR schedule + early stopping on val PR-AUC.

    Signature mirrors ``train_lstm`` for drop-in interchangeability.
    """
    cfg = _cfg()
    window = cfg["features"]["lstm_window"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    train_ds = SlidingWindowDataset(train_df, feature_cols, window)
    val_ds = SlidingWindowDataset(val_df, feature_cols, window)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TCNModel(
        input_size=len(feature_cols),
        num_channels=num_channels,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    stopper = _EarlyStopping(patience=10)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(dev))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_scores.append(probs)
                all_labels.append(yb.numpy())

        y_score = np.concatenate(all_scores)
        y_true = np.concatenate(all_labels)
        val_prauc = float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else 0.0

        if stopper(val_prauc, model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return model.cpu()


def predict_proba_tcn(
    model: TCNModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 30,
    batch_size: int = 512,
) -> np.ndarray:
    """Return probability of failure within X cycles for each windowed sample.

    Signature mirrors ``predict_proba_lstm`` for drop-in interchangeability.
    """
    ds = SlidingWindowDataset(df, feature_cols, window, label_col="label_within_x")
    if len(ds) == 0:
        return np.array([], dtype=np.float32)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    scores = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb)
            scores.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(scores)
