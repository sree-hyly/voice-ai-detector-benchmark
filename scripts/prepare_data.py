"""Build the test set in data/clips/{real_phone,ai_clean,ai_phone}/ plus data/clips/manifest.csv.

real_phone : real people on real phone calls (CallHome English, TalkBank).
ai_clean   : AI speech as produced (ElevenLabs v4, ElevenLabs multilingual v2, Microsoft Edge neural TTS).
ai_phone   : the same AI clips pushed through a simulated phone line (8 kHz, 300-3400 Hz, mu-law),
             so the AI-vs-human comparison isn't just "phone audio vs studio audio".
"""
import asyncio
import json
import random
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vad_bench.audio import load_audio, loudest_window, phone_channel, save_wav  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "data" / "downloads"
OUT = ROOT / "data" / "clips"
SECONDS = 6
N_REAL = 40
random.seed(0)

EDGE_LINES = [
    ("en-US-AriaNeural", "Hi, this is Aria calling from Riverside Dental. I'm just confirming your cleaning appointment for Thursday at two thirty. Does that still work for you?"),
    ("en-US-GuyNeural", "Hello, I'm reaching out from your bank's fraud team. We noticed an unusual charge on your card this morning, and I wanted to check it with you."),
    ("en-US-JennyNeural", "Thanks for calling. I can help you reschedule your delivery. Could you please tell me the order number printed on your receipt?"),
    ("en-GB-SoniaNeural", "Good afternoon, I'm calling about the quote you requested for your home insurance. Have you got a couple of minutes to go through it?"),
    ("en-US-ChristopherNeural", "Hey, it's Chris from the leasing office. Just a quick reminder that maintenance will be in your unit tomorrow between nine and noon."),
    ("en-AU-NatashaNeural", "Hi there, your prescription is ready for pickup at the pharmacy. We're open until eight tonight if that's easier for you."),
    ("en-US-EricNeural", "Um, yeah, so I was just wondering if you guys had any tables free for Saturday night, maybe around seven, for four people?"),
    ("en-IN-NeerjaNeural", "Hello, this is a courtesy call regarding your internet service. We've upgraded your area and wanted to make sure everything is working fine."),
]


def write(kind, name, x, sr, source, rows):
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.wav"
    save_wav(p, x, sr)
    rows.append({"file": str(p.relative_to(ROOT)), "group": kind, "is_ai": int(kind != "real_phone"),
                 "source": source, "seconds": round(len(x) / sr, 2)})
    if kind == "ai_clean":  # every clean AI clip also gets a phone-line twin
        y, ysr = phone_channel(x, sr)
        write("ai_phone", name, y, ysr, source + " (via phone line)", rows)


def load_bytes(data, sr):
    tmp = OUT / "_tmp.audio"
    tmp.write_bytes(data)
    try:
        return load_audio(tmp, sr=sr)
    finally:
        tmp.unlink()


def real_phone(rows):
    t = pq.read_table(DL / "callhome/en/test-00000-of-00001.parquet", columns=["audio", "call_id"]).to_pylist()
    for r in random.sample(t, N_REAL):
        x = load_bytes(r["audio"]["bytes"], 8000)
        write("real_phone", f"callhome_{r['call_id']}", loudest_window(x, 8000, SECONDS), 8000,
              "CallHome English (real phone call)", rows)


def elevenlabs_v4(rows, n=20):
    with tarfile.open(DL / "elevenlabs_v4/raw/audio-00001.tar") as tf:  # read members only, never extract
        mp3s = sorted((m for m in tf.getmembers() if "-en-" in m.name and m.name.endswith(".mp3")), key=lambda m: m.name)
        for m in mp3s[:n]:
            x = load_bytes(tf.extractfile(m).read(), 16000)
            write("ai_clean", "el4_" + Path(m.name).stem[:30], loudest_window(x, 16000, SECONDS), 16000,
                  "ElevenLabs v4", rows)


def elevenlabs_v2(rows, n=6):
    t = pq.read_table(DL / "elevenlabs_v2/data/test-00000-of-00001.parquet", columns=["audio"]).to_pylist()
    for i, r in enumerate(t[:n]):
        x = load_bytes(r["audio"]["bytes"], 16000)
        write("ai_clean", f"el2_{i:02d}", loudest_window(x, 16000, SECONDS), 16000, "ElevenLabs multilingual v2", rows)


async def edge_tts(rows):
    import edge_tts as et

    for voice, text in EDGE_LINES:
        tmp = OUT / "_tmp.mp3"
        await et.Communicate(text, voice).save(str(tmp))
        x = load_audio(tmp, sr=16000); tmp.unlink()
        write("ai_clean", "edge_" + voice, loudest_window(x, 16000, SECONDS), 16000, "Microsoft Edge neural TTS", rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    real_phone(rows); print("real done")
    elevenlabs_v4(rows); print("el4 done")
    elevenlabs_v2(rows); print("el2 done")
    asyncio.run(edge_tts(rows)); print("edge done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "manifest.csv", index=False)
    print(df.groupby(["group", "source"]).size())
