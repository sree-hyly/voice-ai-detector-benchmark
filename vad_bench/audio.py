"""Audio helpers: loading, picking a speech window, and simulating a phone line."""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

MU = 255.0


def load_audio(path, sr: int = 16000) -> np.ndarray:
    """Load any audio file as mono float32 at `sr` Hz."""
    import librosa

    x, _ = librosa.load(str(path), sr=sr, mono=True)
    return x.astype(np.float32)


def save_wav(path, x: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.clip(x, -1, 1), sr, subtype="PCM_16")


def loudest_window(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Return the `seconds`-long stretch with the most energy (i.e. most speech).

    Audio shorter than the window comes back whole, never padded.
    """
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    n = int(seconds * sr)
    if len(x) <= n:
        return x
    energy = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    hop = max(1, sr // 10)
    starts = np.arange(0, len(x) - n + 1, hop)
    best = int(starts[np.argmax(energy[starts + n] - energy[starts])])
    return x[best : best + n]


def mulaw_encode(x: np.ndarray) -> np.ndarray:
    """G.711-style mu-law companding to 8-bit codes (what landline calls carry)."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    return np.round((y + 1) / 2 * MU).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    y = codes.astype(np.float32) / MU * 2 - 1
    return (np.sign(y) * ((1 + MU) ** np.abs(y) - 1) / MU).astype(np.float32)


def phone_channel(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Simulate a narrowband phone call: 8 kHz, 300-3400 Hz band, 8-bit mu-law."""
    y = resample_poly(x.astype(np.float64), 8000, sr) if sr != 8000 else x.astype(np.float64)
    sos = butter(6, [300, 3400], btype="bandpass", fs=8000, output="sos")
    y = sosfiltfilt(sos, y)
    y = mulaw_decode(mulaw_encode(np.clip(y, -1, 1).astype(np.float32)))
    return np.clip(y, -1.0, 1.0).astype(np.float32), 8000
