"""Charts for hard set 3 (voice clones): per-model scatter, Pellav2-vs-VeriSpoof scatter, summary bars."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vad_bench.combo import fit_combo  # noqa: E402
from vad_bench.cutoff import strict_cutoff  # noqa: E402

HUMAN, AI = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OUTD = ROOT / "results/plots"


def style(ax, title, sub):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=26)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5)


def load():
    def pairs(path):
        s = pd.read_csv(ROOT / path)
        s = s[s.model.isin(["Pellav2", "VeriSpoof-Plus"])]
        p = s.pivot_table(index="file", columns="model", values="ai_score").rename(
            columns={"Pellav2": "pella", "VeriSpoof-Plus": "veri"})
        return s.drop_duplicates("file").set_index("file").drop(columns=["model", "ai_score"]).join(p).reset_index()
    c = pairs("results/scores_c.csv")
    return pd.concat([c[c.kind != "fleurs"], pairs("results/scores_c_orig.csv")], ignore_index=True)


ORDER = [("Real caller, call-centre", "Real callers\n(call-centre)"),
         ("Real voice that was cloned (original recording)", "Real people\nwho were cloned"),
         ("ElevenLabs v3 clone", "ElevenLabs v3\nclone"), ("Cartesia sonic clone", "Cartesia\nclone"),
         ("Qwen3-TTS Base clone", "Qwen3-TTS\nclone"), ("Chatterbox clone (open source)", "Chatterbox\nclone"),
         ("F5-TTS clone (open source)", "F5-TTS\nclone"), ("VoxCPM2 clone (open source)", "VoxCPM2\nclone")]


def per_model(d, col, name, fname):
    cut = strict_cutoff(d[col], d.is_ai, 0.01)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for i, (src, _) in enumerate(ORDER):
        g = d[d.source.str.startswith(src)]
        ai = g.is_ai.iloc[0] == 1
        ax.scatter(i + rng.uniform(-0.3, 0.3, len(g)), g[col], s=10, marker="^" if ai else "o",
                   color=AI if ai else HUMAN, alpha=0.55, linewidth=0, zorder=3)
    ax.axvline(1.5, color=GRID, lw=1)
    ax.axhline(cut, color=INK, lw=1.5, ls="--", zorder=2)
    ax.text(ax.get_xlim()[1], cut, f"  cut-off: ≤1% of\n  real people flagged ({cut:.2f})", va="center", color=INK, fontsize=8)
    ax.set_xticks(range(len(ORDER)), [o[1] for o in ORDER], fontsize=8.5, color=INK)
    ax.set_ylabel("AI score (above the line = called AI)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    style(ax, f"{name} on voice clones (set 3): every clip, all 5 call conditions",
          "Circles = real people · Triangles = voice clones of real people. "
          "Line picked on these same clips, so this view is in-sample.")
    handles = [plt.Line2D([], [], ls="", marker="o", color=HUMAN, label="real person"),
               plt.Line2D([], [], ls="", marker="^", color=AI, label="voice clone")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUTD / fname, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def two_d(d):
    model = fit_combo("learned weighting", d)
    s = model(d)
    cut = strict_cutoff(s, d.is_ai, 0.01)
    fig, ax = plt.subplots(figsize=(8, 6.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    hu, ai = d[d.is_ai == 0], d[d.is_ai == 1]
    ax.scatter(hu.pella, hu.veri, s=10, marker="o", color=HUMAN, alpha=0.5, linewidth=0, label="real person", zorder=3)
    ax.scatter(ai.pella, ai.veri, s=12, marker="^", color=AI, alpha=0.5, linewidth=0, label="voice clone", zorder=3)
    gx, gy = np.meshgrid(np.linspace(d.pella.min() - 0.5, d.pella.max() + 0.5, 300),
                         np.linspace(d.veri.min() - 0.5, d.veri.max() + 0.5, 300))
    gs = model(pd.DataFrame({"pella": gx.ravel(), "veri": gy.ravel()})).reshape(gx.shape)
    ax.contour(gx, gy, gs, levels=[cut], colors=INK, linewidths=1.5, linestyles="--", zorder=4)
    pc = strict_cutoff(d.pella, d.is_ai, 0.01)
    vc = strict_cutoff(d.veri, d.is_ai, 0.01)
    ax.axvline(pc, color=MUTED, lw=1, ls=":")
    ax.axhline(vc, color=MUTED, lw=1, ls=":")
    ax.text(pc, ax.get_ylim()[1], " Pellav2 alone", color=MUTED, fontsize=8, va="top")
    ax.text(ax.get_xlim()[0], vc, " VeriSpoof alone", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("Pellav2 AI score", color=MUTED, fontsize=9)
    ax.set_ylabel("VeriSpoof-Plus AI score", color=MUTED, fontsize=9)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    style(ax, "Pellav2 vs VeriSpoof-Plus on voice clones",
          "Dashed = combined (learned weighting); dotted = each model alone. All at ≤1% real people flagged (in-sample).")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUTD / "set3_pellav2_vs_verispoof.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def bars():
    p = pd.read_csv(ROOT / "results/combo_protocol2.csv")
    p = p[p.cutoff_kind == "≤1% humans flagged"].set_index("combo")
    order = ["Pellav2 alone", "VeriSpoof-Plus alone", "either flags", "both must flag", "average", "learned weighting"]
    p = p.loc[order]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(p))[::-1]
    ax.barh(y, p.ai_caught * 100, height=0.55, color=AI, zorder=3)
    ax.errorbar(p.ai_caught * 100, y, xerr=[(p.ai_caught - p.ai_lo) * 100, (p.ai_hi - p.ai_caught) * 100],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    for yi, v in zip(y, p.ai_caught):
        ax.text(1.5, yi, f"{v * 100:.0f}%", va="center", color=SURFACE, fontsize=9, fontweight="bold", zorder=5)
    ax.set_yticks(y, order, fontsize=9, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of voice clones caught (bars: 95% range)", color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    style(ax, "Clones caught when only 1 in 100 real callers is flagged",
          "Set 3, cut-offs tuned on other voices (5-fold, a person and their clone never split).")
    fig.tight_layout()
    fig.savefig(OUTD / "set3_combo_summary.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    per_model(d, "pella", "Pellav2", "set3_pellav2.png")
    per_model(d, "veri", "VeriSpoof-Plus", "set3_verispoof.png")
    two_d(d)
    bars()
    print("ok")
