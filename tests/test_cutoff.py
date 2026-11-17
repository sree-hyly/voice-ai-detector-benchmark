import numpy as np
import pytest

from vad_bench.cutoff import balanced_accuracy, best_balanced_cutoff, strict_cutoff


def test_balanced_accuracy_averages_the_two_rates():
    # 2 of 2 AI caught, 1 of 2 humans passed -> (1.0 + 0.5) / 2
    assert balanced_accuracy(np.array([5, 5, 5, -5]), np.array([1, 1, 0, 0]), 0.0) == 0.75


def test_perfect_split_puts_cutoff_midway_between_groups():
    s = np.array([-4.0, -2.0, 6.0, 8.0])
    y = np.array([0, 0, 1, 1])
    assert best_balanced_cutoff(s, y) == pytest.approx(2.0)  # midway between -2 and 6


def test_cutoff_moves_to_where_the_model_actually_separates():
    # model separates well but around -5, not around its built-in 0
    s = np.array([-9, -8, -7, -3, -2, -1.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    t = best_balanced_cutoff(s, y)
    assert -7 < t < -3
    assert balanced_accuracy(s, y, t) == 1.0


def test_overlap_picks_the_best_achievable_point():
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1])
    t = best_balanced_cutoff(s, y)
    best = max(balanced_accuracy(s, y, c) for c in np.linspace(-1, 2, 3001))
    assert balanced_accuracy(s, y, t) == pytest.approx(best)


def test_constant_scores_do_not_crash():
    t = best_balanced_cutoff(np.zeros(4), np.array([0, 0, 1, 1]))
    assert np.isfinite(t)
    assert balanced_accuracy(np.zeros(4), np.array([0, 0, 1, 1]), t) == 0.5


def test_strict_cutoff_caps_humans_wrongly_flagged():
    humans = np.arange(100, dtype=float)  # scores 0..99
    ai = np.full(10, 200.0)
    s = np.concatenate([humans, ai])
    y = np.array([0] * 100 + [1] * 10)
    t = strict_cutoff(s, y, max_humans_flagged=0.05)
    flagged = np.mean(humans >= t)
    assert flagged <= 0.05
    assert flagged >= 0.04  # and not stricter than needed
    assert np.all(ai >= t)


def test_strict_cutoff_zero_allows_no_human_flags():
    s = np.array([1.0, 2.0, 3.0, 2.5])
    y = np.array([0, 0, 1, 1])
    t = strict_cutoff(s, y, max_humans_flagged=0.0)
    assert t > 2.0
