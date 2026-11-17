"""Turn per-clip scores into per-model results tables."""
from __future__ import annotations

import pandas as pd

from vad_bench.metrics import summarize

# name -> which clip groups are in it. Humans are always the real phone calls.
COMPARISONS = {
    "studio AI vs real phone": ["ai_clean", "real_phone"],
    "phone AI vs real phone": ["ai_phone", "real_phone"],
    "all clips": ["ai_clean", "ai_phone", "real_phone"],
}


def results_table(scores: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    rows = []
    for model, d in scores.groupby("model", sort=False):
        for comp, groups in COMPARISONS.items():
            s = d[d.group.isin(groups)]
            rows.append({"model": model, "comparison": comp, **summarize(s.ai_score, s.is_ai, threshold)})
    return pd.DataFrame(rows)


def per_source_catch_rate(scores: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Share of each AI source's clips that each model called AI."""
    ai = scores[scores.is_ai == 1].assign(caught=lambda d: d.ai_score >= threshold)
    return ai.pivot_table(index="model", columns="source", values="caught", aggfunc="mean", sort=False)


def cross_set_table(set1: pd.DataFrame, set2: pd.DataFrame, set_names=("set 1", "set 2")) -> pd.DataFrame:
    """For each model: pick a cut-off on one set, grade it on the other (both directions).

    Kinds: built-in (0), balanced (best average of AI-caught and humans-passed), and
    strict (lowest cut-off flagging at most 5% of the picking set's humans).
    """
    from vad_bench.cutoff import balanced_accuracy, best_balanced_cutoff, strict_cutoff

    rows = []
    pairs = [(set1, set2, set_names[0], set_names[1]), (set2, set1, set_names[1], set_names[0])]
    for pick, grade, pick_name, grade_name in pairs:
        for model, p in pick.groupby("model", sort=False):
            g = grade[grade.model == model]
            if g.empty:
                continue
            kinds = {
                "built-in": 0.0,
                "balanced": best_balanced_cutoff(p.ai_score, p.is_ai),
                "strict (≤5% humans flagged)": strict_cutoff(p.ai_score, p.is_ai, 0.05),
            }
            for kind, c in kinds.items():
                s = summarize(g.ai_score, g.is_ai, c)
                rows.append({"model": model, "picked_on": pick_name if kind != "built-in" else "-",
                             "graded_on": grade_name, "cutoff_kind": kind, "cutoff": c,
                             "balanced_accuracy": balanced_accuracy(g.ai_score, g.is_ai, c), **s})
    return pd.DataFrame(rows)
