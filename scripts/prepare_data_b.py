"""Build test set 2 in data/clips_b/ (new sources, none used in set 1).

human_call       : real people in role-played customer-service calls, recorded over a VoIP
                   platform, 14 English accents (AppTek Call-Center Dialogues, CC BY-SA 4.0).
human_call_phone : the same human clips through the simulated phone line.
ai_clean         : current voice-agent TTS engines (Cartesia sonic, OpenAI gpt-4o-mini-tts,
                   ElevenLabs v3, Qwen3-TTS) from the ZTTS1-Eval comparison dataset.
ai_phone         : the same AI clips through the simulated phone line.
"""
import random
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vad_bench.audio import load_audio, loudest_window, phone_channel, save_wav  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "data" / "downloads"
OUT = ROOT / "data" / "clips_b"
SECONDS = 6
PER_ACCENT = 4
PER_ENGINE = 15
random.seed(7)

APPTEK = "apptek-com/apptek_callcenter_dialogues"
ZTTS = "jaeyong2/cartesia-sonic-preview-ztts1-zero-shot"
ENGINES = {
    "audio/en/wavs/": "Cartesia sonic (voice clone)",
    "audio_openai_gpt4o_mini_tts_s150/en/wavs/": "OpenAI gpt-4o-mini-tts",
    "audio_elevenlabs_v3_s150/en/wavs/": "ElevenLabs v3 (voice clone)",
    "audio_qwen3_tts_customvoice_s150/en/wavs/": "Qwen3-TTS (open source)",
}


def write(group, name, x, sr, source, rows, twin):
    d = OUT / group
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.wav"
    save_wav(p, x, sr)
    rows.append({"file": str(p.relative_to(ROOT)), "group": group, "is_ai": int(group.startswith("ai")),
                 "source": source, "seconds": round(len(x) / sr, 2)})
    if twin:
        y, ysr = phone_channel(x, sr)
        write(twin, name, y, ysr, source + " (via phone line)", rows, None)


def humans(rows, files):
    by_accent = {}
    for f in files:
        if f.startswith("test/") and f.endswith(".wav"):
            by_accent.setdefault(f.split("/")[1], []).append(f)
    for accent, fs in sorted(by_accent.items()):
        calls = {}
        for f in fs:  # one channel per call, so every clip is a different speaker
            calls.setdefault(f.rsplit("_channel", 1)[0], []).append(f)
        for call in random.sample(sorted(calls), PER_ACCENT):
            f = random.choice(sorted(calls[call]))
            local = hf_hub_download(APPTEK, f, repo_type="dataset", local_dir=DL / "apptek")
            x = load_audio(local, 16000)
            write("human_call", Path(f).stem, loudest_window(x, 16000, SECONDS), 16000,
                  f"AppTek call-center ({accent})", rows, "human_call_phone")


def ai(rows, files):
    for prefix, label in ENGINES.items():
        wavs = sorted(f for f in files if f.startswith(prefix) and f.endswith(".wav"))
        for f in random.sample(wavs, PER_ENGINE):
            local = hf_hub_download(ZTTS, f, repo_type="dataset", local_dir=DL / "ztts")
            x = load_audio(local, 16000)
            tag = prefix.split("/")[0].replace("audio_", "").replace("audio", "cartesia")
            write("ai_clean", f"{tag}_{Path(f).stem}", loudest_window(x, 16000, SECONDS), 16000, label, rows,
                  "ai_phone")


if __name__ == "__main__":
    api = HfApi()
    rows = []
    humans(rows, api.list_repo_files(APPTEK, repo_type="dataset")); print("humans done", flush=True)
    ai(rows, api.list_repo_files(ZTTS, repo_type="dataset")); print("ai done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "manifest.csv", index=False)
    print(df.groupby(["group"]).size())
    print(df[df.group.isin(["ai_clean"])].groupby("source").size())
