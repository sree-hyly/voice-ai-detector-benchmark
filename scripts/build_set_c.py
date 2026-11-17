"""Hard test set 3, step 3 (after cloners/clone.py): add the local clones, then put every clip
(real and AI alike) through all five call conditions in vad_bench.channels.CHANNELS.

Writes data/clips_c/final/<condition>/<clip>.wav and data/clips_c/manifest.csv.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vad_bench.audio import load_audio, loudest_window, save_wav  # noqa: E402
from vad_bench.channels import CHANNELS, apply_channel  # noqa: E402

OUT = ROOT / "data/clips_c"
SR = 16000
LABEL = {"chatterbox": "Chatterbox clone (open source)", "f5": "F5-TTS clone (open source)",
         "voxcpm": "VoxCPM2 clone (open source)"}

base = pd.read_csv(OUT / "base_manifest.csv")
jobs = pd.read_csv(OUT / "clone_jobs.csv")
missing = [j for j in jobs.out if not (ROOT / j).exists()]
if missing:
    sys.exit(f"{len(missing)} clones not made yet, e.g. {missing[0]}")
clones = jobs.assign(file=jobs.out, is_ai=1, source=jobs.engine.map(LABEL), kind="local clone")
allc = pd.concat([base, clones[["file", "is_ai", "source", "kind", "voice_id"]]], ignore_index=True)

noise = load_audio(OUT / "babble.wav", SR)
rows = []
for r in allc.itertuples():
    x = loudest_window(load_audio(ROOT / r.file, SR), SR, 6)
    for ch in CHANNELS:
        y, sr = apply_channel(ch, x, SR, noise=noise)
        slug = re.sub(r"[^a-z0-9]+", "_", ch.lower()).strip("_")
        p = OUT / "final" / slug / f"{'ai' if r.is_ai else 'human'}_{Path(r.file).stem}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        save_wav(p, y, sr)
        rows.append({"file": str(p.relative_to(ROOT)), "group": f"{'AI' if r.is_ai else 'human'} | {ch}",
                     "is_ai": r.is_ai, "source": r.source, "kind": r.kind, "voice_id": r.voice_id, "channel": ch})
m = pd.DataFrame(rows)
m.to_csv(OUT / "manifest.csv", index=False)
print(m.groupby(["is_ai", "kind"]).size().to_string())
print(len(m), "clips;", m.voice_id.nunique(), "voices")
