"""Choosing a model's cut-off (the AI score above which a clip is called AI).

Models ship with a built-in cut-off of 0 (their own 50/50 point), but our results show some
models separate AI from human well while sitting on the wrong side of 0. These pick a cut-off
from labelled clips; always grade the chosen cut-off on different clips than it was picked on.
"""
from __future__ import annotations

import numpy as np


def balanced_accuracy(scores: np.ndarray, is_ai: np.ndarray, cutoff: float) -> float:
    """Average of (share of AI caught) and (share of humans passed): fair when group sizes differ."""
    scores, is_ai = np.asarray(scores, float), np.asarray(is_ai, int)
    called_ai = scores >= cutoff
    return float((called_ai[is_ai == 1].mean() + (~called_ai[is_ai == 0]).mean()) / 2)


def _candidates(scores: np.ndarray) -> np.ndarray:
    u = np.unique(scores)
    mids = (u[:-1] + u[1:]) / 2
    return np.concatenate([[u[0] - 1.0], mids, [u[-1] + 1.0]])


def best_balanced_cutoff(scores, is_ai) -> float:
    """Cut-off with the highest balanced accuracy; placed midway between neighbouring scores."""
    scores = np.asarray(scores, float)
    cands = _candidates(scores)
    vals = np.array([balanced_accuracy(scores, is_ai, c) for c in cands])
    best = np.flatnonzero(vals == vals.max())
    return float(cands[best[len(best) // 2]])  # middle of any tied plateau


def strict_cutoff(scores, is_ai, max_humans_flagged: float = 0.05) -> float:
    """Lowest cut-off that wrongly flags at most `max_humans_flagged` of the humans."""
    scores, is_ai = np.asarray(scores, float), np.asarray(is_ai, int)
    humans = scores[is_ai == 0]
    for c in _candidates(scores):
        if np.mean(humans >= c) <= max_humans_flagged:
            return float(c)
    return float(scores.max() + 1.0)
