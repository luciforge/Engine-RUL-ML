"""PyTorch LSTM for binary failure classification on CMAPSS sliding windows.

Architecture:
    - Bidirectional 2-layer LSTM (hidden=64)
    - Cosine LR schedule
    - Early stopping on validation PR-AUC (patience=10)
    - SlidingWindowDataset: window=30 cycles per sample

Training uses the last label in each window as the target.
Inference returns per-row risk scores by sliding a window of length `window`
over each engine's sorted cycle sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _cfg() -> dict:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


class SlidingWindowDataset(Dataset):
    """Sliding-window dataset over per-engine sorted time series."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        window: int = 30,
        label_col: str = "label_within_x",
    ):
        self.feature_cols = feature_cols
        self.window = window
        self.label_col = label_col
        self._samples: list[tuple[np.ndarray, int]] = []
        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        for _, grp in df.groupby("unit_id"):
            grp = grp.sort_values("cycle").reset_index(drop=True)
            X = grp[self.feature_cols].fillna(0.0).values.astype(np.float32)
            y = grp[self.label_col].values.astype(np.int64)
            for end in range(self.window, len(grp) + 1):
                self._samples.append((X[end - self.window : end], int(y[end - 1])))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        x, y = self._samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class _EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.best_score: float = -np.inf
        self.counter: int = 0
        self.best_state: dict | None = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    device: str | None = None,
) -> LSTMModel:
    """Train LSTM with cosine LR schedule + early stopping on val PR-AUC."""
    cfg = _cfg()
    window = cfg["features"]["lstm_window"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    train_ds = SlidingWindowDataset(train_df, feature_cols, window)
    val_ds = SlidingWindowDataset(val_df, feature_cols, window)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_size=len(feature_cols), hidden_size=hidden_size, num_layers=num_layers
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

        # Validation PR-AUC
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
        if y_true.sum() > 0:
            val_prauc = average_precision_score(y_true, y_score)
        else:
            val_prauc = 0.0

        if stopper(val_prauc, model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return model.cpu()


def predict_proba_lstm(
    model: LSTMModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 30,
    batch_size: int = 512,
) -> np.ndarray:
    """Return probability of failure within X cycles for each windowed sample in df.

    Note: the output has len(SlidingWindowDataset(df, ...)) rows,
    which equals Σ max(0, max_cycle_per_unit - window + 1) across engines.
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
