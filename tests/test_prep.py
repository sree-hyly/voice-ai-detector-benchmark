import numpy as np
import pytest
import torch

from vad_bench.prep import ai_log_odds, first_n_tiled, first_n_zero_padded, standardize


class TestFirstNTiled:
    def test_long_clip_is_cut_to_first_n(self):
        x = np.arange(10, dtype=np.float32)
        np.testing.assert_array_equal(first_n_tiled(x, 4), [0, 1, 2, 3])

    def test_short_clip_is_repeated_not_zero_padded(self):
        x = np.array([1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(first_n_tiled(x, 7), [1, 2, 3, 1, 2, 3, 1])

    def test_empty_clip_gives_silence(self):
        np.testing.assert_array_equal(first_n_tiled(np.zeros(0, np.float32), 3), [0, 0, 0])


class TestFirstNZeroPadded:
    def test_short_clip_is_padded_at_the_end(self):
        x = np.array([1, 2], dtype=np.float32)
        np.testing.assert_array_equal(first_n_zero_padded(x, 4), [1, 2, 0, 0])

    def test_long_clip_is_cut(self):
        np.testing.assert_array_equal(first_n_zero_padded(np.arange(5, dtype=np.float32), 2), [0, 1])


def test_standardize_gives_zero_mean_unit_variance():
    x = np.random.default_rng(0).normal(3, 5, 1000).astype(np.float32)
    y = standardize(x)
    assert abs(y.mean()) < 1e-5 and abs(y.std() - 1) < 1e-3


def test_standardize_silence_does_not_blow_up():
    assert np.all(np.isfinite(standardize(np.zeros(100, np.float32))))


class TestAiLogOdds:
    def test_positive_when_fake_class_wins(self):
        logits = torch.tensor([[3.0, 1.0]])  # class 0 is fake here
        assert ai_log_odds(logits, fake_index=0).item() == pytest.approx(2.0)

    def test_negative_when_real_class_wins(self):
        logits = torch.tensor([[3.0, 1.0]])  # now class 1 is fake, class 0 real
        assert ai_log_odds(logits, fake_index=1).item() == pytest.approx(-2.0)

    def test_zero_matches_fifty_fifty_softmax(self):
        logits = torch.tensor([[0.7, 0.7]])
        assert ai_log_odds(logits, fake_index=0).item() == 0.0

    def test_batch_shape(self):
        out = ai_log_odds(torch.randn(5, 2), fake_index=1)
        assert out.shape == (5,)

    def test_rejects_bad_index(self):
        with pytest.raises(ValueError):
            ai_log_odds(torch.randn(1, 2), fake_index=2)
