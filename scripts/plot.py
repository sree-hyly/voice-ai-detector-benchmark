"""One scatterplot per model: every clip's AI score, grouped by clip type, with cut-off lines.

  python scripts/plot.py "Pellav2" "VeriSpoof-Plus" ...
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HUMAN, AI = "#2a78d6", "#eb6834"  # validated pair (colour-blind safe); shapes differ too
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLUMNS = [  # (set, group, label)
    ("a", "real_phone", "Set 1 humans\nlandline calls"),
    ("b", "human_call", "Set 2 humans\nVoIP calls"),
    ("b", "human_call_phone", "Set 2 humans\nvia phone line"),
    ("a", "ai_clean", "Set 1 AI\nas made"),
    ("a", "ai_phone", "Set 1 AI\nvia phone line"),
    ("b", "ai_clean", "Set 2 AI\nas made"),
    ("b", "ai_phone", "Set 2 AI\nvia phone line"),
]


def plot(model, scores, cut):
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for i, (st, grp, _) in enumerate(COLUMNS):
        d = scores[(scores.set == st) & (scores.group == grp)]
        is_ai = grp.startswith("ai")
        xs = i + rng.uniform(-0.28, 0.28, len(d))
        ax.scatter(xs, d.ai_score, s=26, marker="^" if is_ai else "o", color=AI if is_ai else HUMAN,
                   edgecolor=SURFACE, linewidth=0.8, alpha=0.9, zorder=3)
    ax.axvline(2.5, color=GRID, lw=1, zorder=1)
    lo, hi = ax.get_xlim()
    ax.axhline(0, color=MUTED, lw=1.2, ls=":", zorder=2)
    ax.text(hi, 0, "  built-in cut-off (0)", va="center", ha="left", color=MUTED, fontsize=8.5)
    ax.axhline(cut, color=INK, lw=1.5, ls="--", zorder=2)
    ax.text(hi, cut, f"  tuned cut-off ({cut:.2f})", va="center", ha="left", color=INK, fontsize=8.5)
    ax.set_xticks(range(len(COLUMNS)), [c[2] for c in COLUMNS], fontsize=8.5, color=INK)
    ax.set_ylabel("AI score  (above the line = called AI)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{model}: AI score of every clip", loc="left", color=INK, fontsize=12, pad=26)
    ax.text(0, 1.02, "Circles = real people · Triangles = AI voices. Good model: circles below the dashed line, "
            "triangles above.", transform=ax.transAxes, color=MUTED, fontsize=8.5)
    handles = [plt.Line2D([], [], ls="", marker="o", color=HUMAN, label="real person"),
               plt.Line2D([], [], ls="", marker="^", color=AI, label="AI voice")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    out = ROOT / "results/plots" / (model.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    a = pd.read_csv(ROOT / "results/scores.csv").assign(set="a")
    b = pd.read_csv(ROOT / "results/scores_b.csv").assign(set="b")
    s = pd.concat([a, b])
    cuts = pd.read_csv(ROOT / "results/recommended_cutoffs.csv").set_index("model").balanced_cutoff
    for m in sys.argv[1:]:
        print(plot(m, s[s.model == m], float(cuts[m])))
