"""Input shaping and score direction shared by the model runners.

Every runner reports an "AI score" = log-odds that the clip is AI: positive -> the model says AI,
negative -> the model says human, 0 = the model's own 50/50 point.
"""
from __future__ import annotations

import numpy as np
import torch


def first_n_tiled(x: np.ndarray, n: int) -> np.ndarray:
    """First n samples; short clips repeat themselves (what AASIST-family eval code does)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return np.zeros(n, np.float32)
    if len(x) >= n:
        return x[:n].copy()
    return np.tile(x, n // len(x) + 1)[:n]


def first_n_zero_padded(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros(n, np.float32)
    out[: min(n, len(x))] = x[:n]
    return out


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-5)).astype(np.float32)


def ai_log_odds(logits: torch.Tensor, fake_index: int) -> torch.Tensor:
    """Two-class logits -> logit(fake) - logit(real)."""
    if fake_index not in (0, 1):
        raise ValueError("fake_index must be 0 or 1")
    logits = logits.float()
    return logits[:, fake_index] - logits[:, 1 - fake_index]
