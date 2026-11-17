"""Clone real callers' voices for the hard test set. Run once per cloning tool, in that tool's venv:

  cloners/.venv-chatterbox/bin/python cloners/clone.py chatterbox
  cloners/.venv-f5/bin/python         cloners/clone.py f5
  cloners/.venv-voxcpm/bin/python     cloners/clone.py voxcpm

Reads data/clips_c/clone_jobs.csv (reference recording + line to say), writes each clone's wav.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
engine = sys.argv[1]
jobs = [j for j in csv.DictReader(open(ROOT / "data/clips_c/clone_jobs.csv")) if j["engine"] == engine]
dev = "mps" if torch.backends.mps.is_available() else "cpu"

if engine == "chatterbox":
    from chatterbox.tts import ChatterboxTTS

    m = ChatterboxTTS.from_pretrained(device=dev)

    def clone(ref, text):
        return m.generate(text, audio_prompt_path=ref).squeeze(0).cpu().numpy(), m.sr
elif engine == "f5":
    from f5_tts.api import F5TTS

    m = F5TTS(device=dev)

    def clone(ref, text):
        wav, sr, _ = m.infer(ref_file=ref, ref_text="", gen_text=text, show_info=lambda *a: None, seed=1234)
        return np.asarray(wav), sr
elif engine == "voxcpm":
    from voxcpm import VoxCPM

    m = VoxCPM.from_pretrained(load_denoiser=False, optimize=False, device=dev)

    def clone(ref, text):
        return m.generate(text=text, reference_wav_path=ref), m.tts_model.sample_rate
else:
    raise SystemExit(f"unknown engine {engine}")

for j in jobs:
    out = ROOT / j["out"]
    if out.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    wav, sr = clone(str(ROOT / j["ref"]), j["text"])
    sf.write(out, wav, sr)
    print(f"{engine} {out.name} {len(wav) / sr:.1f}s audio in {time.time() - t:.0f}s", flush=True)
print(f"{engine} DONE {len(jobs)}")
