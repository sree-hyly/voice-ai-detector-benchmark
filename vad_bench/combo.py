"""Combining Pellav2 and VeriSpoof-Plus, plus the grading tools for it.

Each combo is fitted on training clips and returns a function: clips -> score (higher = more AI).
Rule-based combos output a probability-like number where 0.5 is each model's own 50/50 point;
any combo can still be given a tuned cut-off afterwards.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def pella_prob(logit) -> np.ndarray:
    """Pellav2 outputs logit(fake) directly (pellav2_infer.py: p_fake = sigmoid(logit))."""
    return _sigmoid(logit)


def veri_prob(score) -> np.ndarray:
    """VeriSpoof-Plus model card: P(spoof) = sigmoid(0.930 * score - 0.872)."""
    return _sigmoid(0.930 * np.asarray(score, float) - 0.872)


def _logreg(d: pd.DataFrame) -> Callable:
    from sklearn.linear_model import LogisticRegression

    X = np.c_[d.pella, d.veri]
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X, d.is_ai)
    return lambda e: m.predict_proba(np.c_[e.pella, e.veri])[:, 1]


COMBOS = {
    "Pellav2 alone": lambda d: (lambda e: np.asarray(e.pella, float)),
    "VeriSpoof-Plus alone": lambda d: (lambda e: np.asarray(e.veri, float)),
    "either flags": lambda d: (lambda e: np.maximum(pella_prob(e.pella), veri_prob(e.veri))),
    "both must flag": lambda d: (lambda e: np.minimum(pella_prob(e.pella), veri_prob(e.veri))),
    "average": lambda d: (lambda e: (pella_prob(e.pella) + veri_prob(e.veri)) / 2),
    "learned weighting": _logreg,
}


def fit_combo(name: str, train: pd.DataFrame) -> Callable[[pd.DataFrame], np.ndarray]:
    return COMBOS[name](train)


def grouped_folds(groups, k: int = 5, seed: int = 0):
    """K folds where every group (voice) is wholly in train or wholly in test."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    out = []
    for f in range(k):
        test_groups = set(uniq[f::k])
        test = np.array([g in test_groups for g in groups])
        out.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return out


def bootstrap_ci(d: pd.DataFrame, stat: Callable[[pd.DataFrame], float], group: str = "voice_id",
                 n: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Confidence range by resampling whole voices (a person and their clone travel together)."""
    rng = np.random.default_rng(seed)
    parts = {g: x for g, x in d.groupby(group)}
    keys = list(parts)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        vals.append(stat(pd.concat([parts[keys[i]] for i in pick])))
    a = (1 - level) / 2
    return float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))


def operating_points(train_scores, train_y, test_scores, test_y, built_in: float) -> pd.DataFrame:
    """Grade three cut-offs on test clips: the built-in one, and two picked on the training clips so
    that at most 5% / 1% of training humans would be flagged."""
    from vad_bench.cutoff import strict_cutoff
    from vad_bench.metrics import summarize

    cuts = {"built-in": built_in,
            "≤5% humans flagged": strict_cutoff(train_scores, train_y, 0.05),
            "≤1% humans flagged": strict_cutoff(train_scores, train_y, 0.01)}
    rows = [{"cutoff_kind": k, "cutoff": c, **summarize(test_scores, test_y, c)} for k, c in cuts.items()]
    return pd.DataFrame(rows)


def cv_predictions(d: pd.DataFrame, combo: str, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold scores: each fold's combo and cut-offs are fitted on the other folds only.
    A voice (person + their clone) is always entirely in one fold."""
    from vad_bench.cutoff import strict_cutoff

    d = d.reset_index(drop=True)
    parts = []
    for f, (tr, te) in enumerate(grouped_folds(d.voice_id.values, k, seed)):
        train, test = d.iloc[tr], d.iloc[te]
        model = fit_combo(combo, train)
        s_tr = model(train)
        parts.append(test.assign(score=model(test), fold=f,
                                 cutoff_5=strict_cutoff(s_tr, train.is_ai, 0.05),
                                 cutoff_1=strict_cutoff(s_tr, train.is_ai, 0.01)))
    return pd.concat(parts).sort_index()
