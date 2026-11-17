"""Results across both test sets.

Writes:
  results/summary_b.csv          set 2 with built-in cut-offs
  results/cross_set.csv          cut-off picked on one set, graded on the other
  results/recommended_cutoffs.csv cut-offs picked on both sets together (for deployment)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vad_bench.cutoff import best_balanced_cutoff, strict_cutoff  # noqa: E402
from vad_bench.metrics import summarize  # noqa: E402
from vad_bench.report import cross_set_table  # noqa: E402

a = pd.read_csv(ROOT / "results/scores.csv")
b = pd.read_csv(ROOT / "results/scores_b.csv")
pct = lambda v: f"{100 * v:.0f}%"

rows = []
for comp, groups in {"set 2: AI vs humans, as recorded": ["ai_clean", "human_call"],
                     "set 2: AI vs humans, both via phone line": ["ai_phone", "human_call_phone"],
                     "set 2: all clips": ["ai_clean", "ai_phone", "human_call", "human_call_phone"]}.items():
    for m, d in b.groupby("model", sort=False):
        s = d[d.group.isin(groups)]
        rows.append({"model": m, "comparison": comp, **summarize(s.ai_score, s.is_ai, 0.0)})
sb = pd.DataFrame(rows)
sb.to_csv(ROOT / "results/summary_b.csv", index=False)
for comp, d in sb.groupby("comparison", sort=False):
    d = d.sort_values("auc", ascending=False)
    print(f"\n== {comp} (built-in cut-off; {int(d.n_ai.iloc[0])} AI, {int(d.n_human.iloc[0])} human)")
    print(pd.DataFrame({"model": d.model, "accuracy": d.accuracy.map(pct), "AI caught": d.ai_caught.map(pct),
                        "humans passed": d.humans_passed.map(pct), "AUC": d.auc.round(2)}).to_string(index=False))

x = cross_set_table(a, b, set_names=("set 1", "set 2"))
x.to_csv(ROOT / "results/cross_set.csv", index=False)
for graded in ["set 2", "set 1"]:
    print(f"\n== graded on {graded} (cut-off picked on the other set)")
    d = x[x.graded_on == graded]
    piv = d.pivot_table(index="model", columns="cutoff_kind", values="balanced_accuracy", sort=False)
    piv = piv[["built-in", "balanced", "strict (≤5% humans flagged)"]].sort_values("balanced", ascending=False)
    print((piv * 100).round(0).astype(int).astype(str).add("%").to_string())

both = pd.concat([a, b])
rec = []
for m, d in both.groupby("model", sort=False):
    rec.append({"model": m, "balanced_cutoff": best_balanced_cutoff(d.ai_score, d.is_ai),
                "strict_cutoff": strict_cutoff(d.ai_score, d.is_ai, 0.05),
                "auc_both_sets": summarize(d.ai_score, d.is_ai, 0.0)["auc"]})
rec = pd.DataFrame(rec).sort_values("auc_both_sets", ascending=False)
rec.to_csv(ROOT / "results/recommended_cutoffs.csv", index=False)
print("\n== cut-offs picked on both sets together")
print(rec.round(2).to_string(index=False))
