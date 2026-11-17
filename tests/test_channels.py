import numpy as np
import pytest
from scipy.signal import welch

from vad_bench.channels import CHANNELS, add_noise, amr_nb, apply_channel, opus


def tone(f, sec=2.0, sr=16000, amp=0.4):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def speechlike(sec=2.0, sr=16000):
    rng = np.random.default_rng(0)
    return (0.2 * rng.standard_normal(int(sec * sr)) * (1 + np.sin(np.arange(int(sec * sr)) / 800))).astype(np.float32)


def band_energy(x, sr, lo, hi):
    f, p = welch(x, sr, nperseg=1024)
    return p[(f >= lo) & (f < hi)].sum()


class TestAmrNb:
    def test_is_narrowband_8khz_and_keeps_length(self):
        y, sr = amr_nb(speechlike(), 16000)
        assert sr == 8000
        assert abs(len(y) - 16000) < 400  # codec frames may pad a little

    def test_actually_changes_the_signal(self):
        x = speechlike()
        y, _ = amr_nb(x, 16000)
        x8 = x[::2][: len(y)]
        assert np.mean((y[: len(x8)] - x8) ** 2) > 1e-4


class TestOpus:
    def test_wideband_16khz_and_length(self):
        y, sr = opus(speechlike(), 16000, kbps=12)
        assert sr == 16000
        assert abs(len(y) - 32000) < 800

    def test_low_bitrate_loses_more_than_high(self):
        x = speechlike()
        lo, _ = opus(x, 16000, kbps=8)
        hi, _ = opus(x, 16000, kbps=32)
        n = min(len(lo), len(hi), len(x))
        assert np.mean((lo[:n] - x[:n]) ** 2) > np.mean((hi[:n] - x[:n]) ** 2)


class TestAddNoise:
    def test_hits_requested_snr(self):
        x = tone(440)
        noise = np.random.default_rng(1).standard_normal(8000).astype(np.float32)
        y = add_noise(x, noise, snr_db=10)
        n = y - x
        snr = 10 * np.log10(np.mean(x ** 2) / np.mean(n ** 2))
        assert snr == pytest.approx(10, abs=0.2)

    def test_short_noise_is_looped_to_cover_the_clip(self):
        x = tone(440)
        y = add_noise(x, np.ones(100, np.float32), snr_db=20)
        assert len(y) == len(x)
        assert np.all(np.abs(y[-50:] - x[-50:]) > 0)

    def test_output_stays_in_range(self):
        y = add_noise(tone(440, amp=0.99), np.ones(10, np.float32), snr_db=0)
        assert np.max(np.abs(y)) <= 1.0


def test_every_channel_runs_and_returns_audio():
    noise = speechlike(3)
    for name in CHANNELS:
        y, sr = apply_channel(name, speechlike(), 16000, noise=noise)
        assert sr in (8000, 16000) and len(y) > 0.9 * 2 * sr and np.all(np.isfinite(y)), name


def test_as_recorded_is_untouched():
    x = speechlike()
    y, sr = apply_channel("as recorded", x, 16000)
    assert sr == 16000 and np.array_equal(x, y)


def test_noisy_channel_needs_noise():
    with pytest.raises(ValueError):
        apply_channel("mobile + background chatter", speechlike(), 16000, noise=None)
