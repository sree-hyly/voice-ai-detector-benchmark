"""Scoring: how well a detector separates AI clips from human clips.

Every score is oriented so that HIGHER = MORE LIKELY AI.
"""
from __future__ import annotations

import math

import numpy as np


def equal_error_rate(scores: np.ndarray, is_ai: np.ndarray) -> float:
    """Error rate at the threshold where missed-AI rate == wrongly-flagged-human rate."""
    ai, hu = scores[is_ai == 1], scores[is_ai == 0]
    if len(ai) == 0 or len(hu) == 0:
        return math.nan
    best = None
    for t in np.unique(np.concatenate([scores, [np.inf]])):
        miss = np.mean(ai < t)  # AI let through
        false_alarm = np.mean(hu >= t)  # human flagged as AI
        gap = abs(miss - false_alarm)
        if best is None or gap < best[0] - 1e-12:
            best = (gap, (miss + false_alarm) / 2)
    return float(best[1])


def auc(scores: np.ndarray, is_ai: np.ndarray) -> float:
    """Chance a random AI clip scores higher than a random human clip (ties count half)."""
    ai, hu = scores[is_ai == 1], scores[is_ai == 0]
    if len(ai) == 0 or len(hu) == 0:
        return math.nan
    diff = ai[:, None] - hu[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def summarize(scores, is_ai, threshold: float) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    is_ai = np.asarray(is_ai, dtype=int)
    if scores.shape != is_ai.shape:
        raise ValueError("scores and labels differ in length")
    called_ai = scores >= threshold
    ai, hu = is_ai == 1, is_ai == 0
    rate = lambda m: float(np.mean(m)) if len(m) else math.nan
    return {
        "n_ai": int(ai.sum()),
        "n_human": int(hu.sum()),
        "accuracy": rate(called_ai == ai),
        "ai_caught": rate(called_ai[ai]),
        "humans_passed": rate(~called_ai[hu]),
        "eer": equal_error_rate(scores, is_ai),
        "auc": auc(scores, is_ai),
    }
