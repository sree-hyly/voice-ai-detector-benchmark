"""Realistic call conditions applied equally to human and AI clips.

Codecs run through ffmpeg (needs libopencore_amrnb and libopus, both in the Homebrew build).
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy.signal import resample_poly

from vad_bench.audio import phone_channel


def _ffmpeg_roundtrip(x: np.ndarray, sr: int, enc_args: list[str], fmt: str, out_sr: int) -> np.ndarray:
    """Encode float audio with ffmpeg, decode it again, return float32 at out_sr."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", *enc_args, "-f", fmt, "pipe:1"],
        input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", fmt, "-i", "pipe:0", "-f", "s16le", "-ar", str(out_sr), "-ac", "1", "pipe:1"],
        input=enc, capture_output=True, check=True).stdout
    return np.frombuffer(dec, "<i2").astype(np.float32) / 32768


def amr_nb(x: np.ndarray, sr: int, bitrate: str = "12.2k") -> tuple[np.ndarray, int]:
    """Mobile-phone voice codec (AMR narrowband, 8 kHz)."""
    x8 = resample_poly(x, 8000, sr).astype(np.float32) if sr != 8000 else x
    return _ffmpeg_roundtrip(x8, 8000, ["-c:a", "libopencore_amrnb", "-b:a", bitrate], "amr", 8000), 8000


def opus(x: np.ndarray, sr: int, kbps: int = 12) -> tuple[np.ndarray, int]:
    """Internet-call codec (Opus voice mode, as in WhatsApp/Teams/WebRTC)."""
    y = _ffmpeg_roundtrip(x, sr, ["-c:a", "libopus", "-b:a", f"{kbps}k", "-application", "voip"], "ogg", 16000)
    return y, 16000


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise under x at the given signal-to-noise ratio (noise looped/cut to x's length)."""
    reps = int(np.ceil(len(x) / len(noise)))
    n = np.tile(noise, reps)[: len(x)].astype(np.float64)
    ps, pn = np.mean(x.astype(np.float64) ** 2), np.mean(n ** 2)
    if pn == 0:
        return x.copy()
    n *= np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    y = x + n
    peak = np.max(np.abs(y))
    if peak > 1:
        y = y / peak
    return y.astype(np.float32)


CHANNELS = ["as recorded", "landline (G.711)", "mobile (AMR-NB)", "internet call (Opus 12k)",
            "mobile + background chatter"]


def apply_channel(name: str, x: np.ndarray, sr: int, noise: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    if name == "as recorded":
        return x, sr
    if name == "landline (G.711)":
        return phone_channel(x, sr)
    if name == "mobile (AMR-NB)":
        return amr_nb(x, sr)
    if name == "internet call (Opus 12k)":
        return opus(x, sr, 12)
    if name == "mobile + background chatter":
        if noise is None:
            raise ValueError("background chatter needs a noise clip")
        return amr_nb(add_noise(x, noise, snr_db=12), sr)
    raise KeyError(name)
