"""Hard set 3, step 4: replace the 'restored' FLEURS-R human recordings with the original recordings.

The commercial clones were made from FLEURS-R references, which an AI restoration model regenerated,
so they are not untouched human audio. For each one we find the original FLEURS recording of the same
sentence (1-3 readers per sentence) whose spectrogram lines up best with the restored version, and put
it through the same five call conditions. Writes data/clips_c/manifest_orig.csv.
"""
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vad_bench.audio import load_audio, loudest_window, save_wav  # noqa: E402
from vad_bench.channels import CHANNELS, apply_channel  # noqa: E402

OUT, SR = ROOT / "data/clips_c", 16000
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def logmel(x):
    return np.log(librosa.feature.melspectrogram(y=x, sr=SR, n_mels=64) + 1e-6)


def match_score(a, b):
    A, B = logmel(a), logmel(b)
    n = min(A.shape[1], B.shape[1])
    return float(np.corrcoef(A[:, :n].ravel(), B[:, :n].ravel())[0, 1]), abs(A.shape[1] - B.shape[1]) / max(A.shape[1], B.shape[1])


base = pd.read_csv(OUT / "base_manifest.csv")
hum = base[base.kind == "fleurs"].copy()
hum["utt"] = hum.file.str.extract(r"(en-uttid_\d+)")[0]
zt = pd.concat([pq.read_table(ROOT / f"data/downloads/ztts_full/comparison_full500/train-0000{i}-of-00020.parquet",
                              columns=["utt", "prompt_text", "audio_reference"]).to_pandas() for i in (0, 1)])
zt = zt[zt.utt.isin(hum.utt)].set_index("utt")
keys = {norm(t) for t in zt.prompt_text}
fl = pq.read_table(ROOT / "data/downloads/fleurs/parquet-data/en_us/train-00000-of-00001.parquet",
                   columns=["raw_transcription", "audio"]).to_pandas()
fl = fl[fl.raw_transcription.map(norm).isin(keys)]
fl["k"] = fl.raw_transcription.map(norm)

tmp = OUT / "_tmp.audio"
noise = load_audio(OUT / "babble.wav", SR)
rows, report = [], []
for r in hum.itertuples():
    tmp.write_bytes(zt.loc[r.utt, "audio_reference"]["bytes"])
    restored = load_audio(tmp, SR)
    cands = fl[fl.k == norm(zt.loc[r.utt, "prompt_text"])]
    scored = []
    for c in cands.itertuples():
        tmp.write_bytes(c.audio["bytes"])
        x = load_audio(tmp, SR)
        scored.append((*match_score(restored, x), x))
    corr, len_gap, orig = max(scored, key=lambda s: s[0])
    report.append({"utt": r.utt, "candidates": len(scored), "best_corr": corr,
                   "runner_up_corr": sorted(s[0] for s in scored)[-2] if len(scored) > 1 else np.nan, "len_gap": len_gap})
    x = loudest_window(orig, SR, 6)
    for ch in CHANNELS:
        y, sr = apply_channel(ch, x, SR, noise=noise)
        slug = re.sub(r"[^a-z0-9]+", "_", ch.lower()).strip("_")
        p = OUT / "final" / slug / f"human_fleurs_original_{r.utt}.wav"
        save_wav(p, y, sr)
        rows.append({"file": str(p.relative_to(ROOT)), "group": f"human | {ch}", "is_ai": 0,
                     "source": "Real voice that was cloned (original recording)", "kind": "fleurs original",
                     "voice_id": r.voice_id, "channel": ch})
tmp.unlink()
pd.DataFrame(rows).to_csv(OUT / "manifest_orig.csv", index=False)
rep = pd.DataFrame(report)
rep.to_csv(OUT / "fleurs_match_report.csv", index=False)
print(rep[["best_corr", "runner_up_corr", "len_gap"]].describe().round(3).to_string())
