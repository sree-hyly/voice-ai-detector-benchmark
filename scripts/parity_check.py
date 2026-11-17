"""Check our runners reproduce the Speech Anti-Spoofing Arena's published per-file scores.

Takes 24 In-the-Wild clips, scores them with our runner, and compares against the Arena's
scores for the same files (which say "higher = more human"). A correctly wired runner gives
rank correlation near +1 with the negated Arena score; a mis-wired one gives ~0 or negative.
"""
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vad_bench.audio import load_audio  # noqa: E402
from vad_bench.runners import all_runners  # noqa: E402

ARENA = {"AASIST": "AASIST", "W2V2-AASIST": "W2V2-AASIST", "ProSDD (ASVspoof2019)": "ProSDD-ASV19",
         "ProSDD (ASVspoof5)": "ProSDD-ASV5", "Pellav2": "Pellav2", "SONAR": "SONAR-Full",
         "NII MMS-1B": "MMS-1B-AntiDeepfake", "NII XLS-R-2B": "XLS-R-2B-AntiDeepfake"}
N = 24

t = pq.read_table(ROOT / "data/downloads/inthewild/data/test-00000-of-00008.parquet").slice(0, 400).to_pylist()
rng = np.random.default_rng(1)
pick = [t[i] for i in rng.choice(len(t), N, replace=False)]
tmp = ROOT / "data/downloads/_parity.audio"
wavs = {}
for r in pick:
    tmp.write_bytes(r["audio"]["bytes"])
    wavs["ITW_" + Path(r["path"]).stem] = load_audio(tmp, 16000)
tmp.unlink()

out = []
for r in all_runners():
    if r.name not in ARENA or (len(sys.argv) > 1 and r.name not in sys.argv[1:]):
        continue
    url = f"https://huggingface.co/SpeechAntiSpoofingBenchmarks/{ARENA[r.name]}/resolve/main/.eval_results/SpeechAntiSpoofingBenchmarks/InTheWild/scores.txt"
    pub = dict(line.split() for line in urllib.request.urlopen(url).read().decode().splitlines() if line.strip())
    ids = [k for k in wavs if k in pub]
    r.load()
    ours = np.array([r.ai_score(wavs[k]) for k in ids])
    r.unload()
    theirs = -np.array([float(pub[k]) for k in ids])
    rho = spearmanr(ours, theirs).statistic
    out.append({"model": r.name, "clips": len(ids), "rank_corr": round(rho, 3)})
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(ROOT / "results/parity_check.csv", index=False)
