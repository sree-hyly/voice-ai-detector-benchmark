"""Score every clip with every model. Results go to results/scores.csv (one row per model x clip).

  python scripts/run_all.py              # all models, skips models already in scores.csv
  python scripts/run_all.py --smoke      # 2 clips per model, prints only
  python scripts/run_all.py --only SONAR
  python scripts/run_all.py --set b      # test set 2
"""
import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vad_bench.audio import load_audio  # noqa: E402
from vad_bench.runners import all_runners  # noqa: E402


ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--only", nargs="*")
ap.add_argument("--set", choices=["a", "b", "c", "c-orig"], default="a", help="a = set 1, b = set 2, c = set 3 (voice clones)")
args = ap.parse_args()
MANIFEST = ROOT / {"a": "data/clips/manifest.csv", "b": "data/clips_b/manifest.csv", "c": "data/clips_c/manifest.csv",
                   "c-orig": "data/clips_c/manifest_orig.csv"}[args.set]
OUT = ROOT / "results" / {"a": "scores.csv", "b": "scores_b.csv", "c": "scores_c.csv", "c-orig": "scores_c_orig.csv"}[args.set]

man = pd.read_csv(MANIFEST)
if args.smoke:
    man = pd.concat([man[man.is_ai == 0].head(1), man[man.is_ai == 1].head(1)])
wavs = {f: load_audio(ROOT / f, sr=16000) for f in man.file}
done = set(pd.read_csv(OUT).model) if OUT.exists() and not args.smoke else set()

for r in all_runners():
    if args.only and r.name not in args.only:
        continue
    if r.name in done:
        print(f"skip {r.name} (already scored)")
        continue
    t0 = time.time()
    try:
        r.load()
        t1 = time.time()
        rows = []
        for row in man.itertuples(index=False):
            rows.append({"model": r.name, **row._asdict(), "ai_score": r.ai_score(wavs[row.file])})
        t2 = time.time()
        df = pd.DataFrame(rows)
        print(f"{r.name}: loaded {t1 - t0:.0f}s, {len(rows)} clips in {t2 - t1:.0f}s on {r.device}")
        if args.smoke:
            print(df[["group", "ai_score"]].to_string(index=False))
        else:
            df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    except Exception:
        print(f"FAILED {r.name}")
        traceback.print_exc()
    finally:
        r.unload()
        gc.collect()
