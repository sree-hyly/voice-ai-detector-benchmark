import numpy as np
import pytest

from vad_bench.audio import loudest_window, mulaw_decode, mulaw_encode, phone_channel


def tone(freq, seconds, sr, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestLoudestWindow:
    def test_picks_the_loud_part_of_mostly_silent_audio(self):
        sr = 16000
        x = np.zeros(sr * 20, dtype=np.float32)
        x[sr * 12 : sr * 15] = tone(440, 3, sr)  # speech-like burst at 12-15 s
        w = loudest_window(x, sr, seconds=4)
        assert len(w) == sr * 4
        # the whole 3-second burst must be inside the chosen window
        assert np.sum(np.abs(w) > 0.1) >= 0.95 * np.sum(np.abs(x) > 0.1)

    def test_short_audio_is_returned_whole_not_padded(self):
        sr = 16000
        x = tone(440, 2, sr)
        w = loudest_window(x, sr, seconds=8)
        assert len(w) == len(x)
        np.testing.assert_array_equal(w, x)

    def test_rejects_non_positive_window(self):
        with pytest.raises(ValueError):
            loudest_window(np.zeros(100, dtype=np.float32), 16000, seconds=0)


class TestMulaw:
    def test_round_trip_is_close_but_lossy(self):
        x = np.linspace(-0.9, 0.9, 1001).astype(np.float32)
        codes = mulaw_encode(x)
        assert codes.dtype == np.uint8
        assert len(np.unique(codes)) <= 256
        y = mulaw_decode(codes)
        err = np.abs(y - x)
        assert err.max() < 0.05  # 8-bit companding error stays small
        assert err.max() > 0  # but it is genuinely quantised, not a passthrough

    def test_quiet_signals_keep_more_relative_precision_than_loud(self):
        quiet = np.full(10, 0.01, dtype=np.float32)
        loud = np.full(10, 0.9, dtype=np.float32)
        rel_q = abs(mulaw_decode(mulaw_encode(quiet))[0] - 0.01) / 0.01
        rel_l = abs(mulaw_decode(mulaw_encode(loud))[0] - 0.9) / 0.9
        assert rel_q < 0.1 and rel_l < 0.1


class TestPhoneChannel:
    def test_output_is_8khz_and_same_duration(self):
        sr = 16000
        x = tone(1000, 2, sr)
        y, out_sr = phone_channel(x, sr)
        assert out_sr == 8000
        assert abs(len(y) - 16000) <= 2

    def test_removes_high_frequencies_and_keeps_voice_band(self):
        sr = 16000
        _, ysr = phone_channel(tone(1000, 1, sr), sr)
        voice, _ = phone_channel(tone(1000, 1, sr), sr)
        hiss, _ = phone_channel(tone(6000, 1, sr), sr)  # above the 3.4 kHz phone band
        rumble, _ = phone_channel(tone(60, 1, sr), sr)  # below the 300 Hz phone band
        rms = lambda a: float(np.sqrt(np.mean(a[200:-200] ** 2)))
        assert rms(voice) > 0.25
        assert rms(hiss) < 0.02
        assert rms(rumble) < 0.1

    def test_output_is_clipped_to_valid_range(self):
        y, _ = phone_channel(tone(1000, 1, 16000, amp=3.0), 16000)
        assert np.all(np.abs(y) <= 1.0)
