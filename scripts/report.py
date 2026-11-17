"""Print and save the results tables from results/scores.csv."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vad_bench.report import per_source_catch_rate, results_table  # noqa: E402

s = pd.read_csv(ROOT / "results/scores.csv")
t = results_table(s)
t.to_csv(ROOT / "results/summary.csv", index=False)
pct = lambda v: f"{100 * v:.0f}%"
for comp in t.comparison.unique():
    d = t[t.comparison == comp].sort_values("auc", ascending=False)
    print(f"\n== {comp}  ({int(d.n_ai.iloc[0])} AI clips, {int(d.n_human.iloc[0])} human clips)")
    print(pd.DataFrame({
        "model": d.model,
        "accuracy": d.accuracy.map(pct),
        "AI caught": d.ai_caught.map(pct),
        "humans passed": d.humans_passed.map(pct),
        "best-threshold error": d.eer.map(pct),
        "separation (AUC)": d.auc.round(2),
    }).to_string(index=False))
c = per_source_catch_rate(s)
c.to_csv(ROOT / "results/catch_by_source.csv")
print("\n== share of each AI source called AI")
print((c * 100).round(0).astype("Int64").to_string())
