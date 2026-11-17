"""Hard test set 3, step 1: real voices, commercial clones of those voices, and clone jobs.

Real people
  fleurs : the real recordings that Cartesia / ElevenLabs v3 / Qwen3-TTS cloned (read speech).
  apptek : new call-centre callers (none used in set 2); a 6 s clip of each caller.
AI clones
  commercial : Cartesia sonic, ElevenLabs v3 (instant clone), Qwen3-TTS Base clones of the fleurs voices.
  local      : written as jobs here; scripts in cloners/ clone each apptek caller from a *different*
               10 s stretch of their own call, saying customer-service lines (see LINES).
Every clip carries `voice_id`: a real clip and the clone of that same voice share it, so later
cross-validation can keep a voice entirely on one side.
"""
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vad_bench.audio import load_audio, loudest_window, save_wav  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "data" / "downloads"
OUT = ROOT / "data" / "clips_c"
SR, SECONDS = 16000, 6
random.seed(11)
APPTEK = "apptek-com/apptek_callcenter_dialogues"
CLONE_ENGINES = ["chatterbox", "f5", "voxcpm"]
LINES = [
    "Hi, yes, I'm calling about a charge on my statement that I don't recognise.",
    "I'd like to dispute a transaction from last Tuesday. It was about two hundred and forty dollars.",
    "Can you tell me when my payment is due this month? I think I missed the email.",
    "I'm travelling to Spain next week and I wanted to let you know so my card doesn't get blocked.",
    "I think my card was stolen. I haven't been able to find it since this morning.",
    "Okay, great. And will the new card come in the mail, or do I need to pick it up somewhere?",
    "No, that's not me. I've never shopped at that store, I don't even know where it is.",
    "Could you increase my credit limit? I've been a customer for about six years now.",
    "Sure, my date of birth is the fourteenth of March, nineteen eighty-two.",
    "Yeah, hi, I just need to update the mailing address on my account, we moved last month.",
    "Um, so I got a text saying there was a problem with my account, and I wanted to check if it's real.",
    "Can you send me the statement for October? I need it for my taxes.",
]


def save(group, name, x, source, voice_id, rows, kind):
    d = OUT / "base" / group
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.wav"
    save_wav(p, x, SR)
    rows.append({"file": str(p.relative_to(ROOT)), "is_ai": int(group == "ai"), "source": source,
                 "kind": kind, "voice_id": voice_id})


def fleurs_and_commercial(rows):
    used = set(pd.read_csv(ROOT / "data/clips_b/manifest.csv").file.str.extract(r"(uttid_\d+)")[0].dropna())
    cols = ["utt", "prompt_text", "audio_reference", "audio_cartesia", "audio_elevenlabs", "audio_qwen3base"]
    t = pd.concat([pq.read_table(DL / f"ztts_full/comparison_full500/train-0000{i}-of-00020.parquet",
                                 columns=cols).to_pandas() for i in (0, 1)])
    t = t[~t.utt.str.replace("en-", "").isin(used)].drop_duplicates("prompt_text")
    t = t.sample(90, random_state=3).reset_index(drop=True)
    engines = [("audio_cartesia", "Cartesia sonic clone"), ("audio_elevenlabs", "ElevenLabs v3 clone"),
               ("audio_qwen3base", "Qwen3-TTS Base clone")]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_tmp.audio"
    for i, r in t.iterrows():
        vid = "fleurs:" + hashlib.md5(r.prompt_text.encode()).hexdigest()[:10]
        tmp.write_bytes(r.audio_reference["bytes"])
        save("human", f"fleurs_{r.utt}", loudest_window(load_audio(tmp, SR), SR, SECONDS),
             "Real voice that was cloned (read speech)", vid, rows, "fleurs")
        col, label = engines[i % 3]
        tmp.write_bytes(r[col]["bytes"])
        save("ai", f"{col.split('_')[1]}_{r.utt}", loudest_window(load_audio(tmp, SR), SR, SECONDS),
             label, vid, rows, "commercial clone")
    tmp.unlink()


def apptek(rows, jobs):
    used = set(pd.read_csv(ROOT / "data/clips_b/manifest.csv").file.map(lambda f: Path(f).stem.rsplit("_channel", 1)[0]))
    files = [f for f in HfApi().list_repo_files(APPTEK, repo_type="dataset") if f.startswith("test/") and f.endswith(".wav")]
    by_acc = {}
    for f in files:
        call = Path(f).stem.rsplit("_channel", 1)[0]
        if call not in used:
            by_acc.setdefault(f.split("/")[1], {}).setdefault(call, []).append(f)
    k = 0
    babble_src = []
    for acc, calls in sorted(by_acc.items()):
        picks = random.sample(sorted(calls), 5)
        for j, call in enumerate(picks):
            f = random.choice(sorted(calls[call]))
            x = load_audio(hf_hub_download(APPTEK, f, repo_type="dataset", local_dir=DL / "apptek"), SR)
            if j == 4:  # fifth caller per accent only feeds the background-chatter noise
                babble_src.append(loudest_window(x, SR, 20))
                continue
            half = len(x) // 2  # reference from one half of the call, test clip from the other
            ref_part, test_part = (x[:half], x[half:]) if random.random() < 0.5 else (x[half:], x[:half])
            vid = "apptek:" + call
            save("human", Path(f).stem, loudest_window(test_part, SR, SECONDS),
                 f"Real caller, call-centre ({acc})", vid, rows, "apptek")
            ref_path = OUT / "refs" / f"{Path(f).stem}.wav"
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            save_wav(ref_path, loudest_window(ref_part, SR, 10), SR)
            eng = CLONE_ENGINES[k % 3]
            jobs.append({"engine": eng, "voice_id": vid, "ref": str(ref_path.relative_to(ROOT)),
                         "text": LINES[k % len(LINES)],
                         "out": str((OUT / "clones" / eng / f"{Path(f).stem}.wav").relative_to(ROOT))})
            k += 1
    m = min(len(b) for b in babble_src)
    babble = np.sum([b[:m] for b in babble_src[:6]], axis=0)
    save_wav(OUT / "babble.wav", babble / np.max(np.abs(babble)) * 0.5, SR)


if __name__ == "__main__":
    rows, jobs = [], []
    fleurs_and_commercial(rows); print("fleurs + commercial done", flush=True)
    apptek(rows, jobs); print("apptek done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "base_manifest.csv", index=False)
    pd.DataFrame(jobs).to_csv(OUT / "clone_jobs.csv", index=False)
    print(pd.DataFrame(rows).groupby(["is_ai", "source"]).size().to_string())
    print(pd.DataFrame(jobs).engine.value_counts().to_string())
