import math

import numpy as np
import pytest

from vad_bench.metrics import equal_error_rate, summarize


def test_perfect_separation_scores_full_marks():
    # higher score = more likely AI
    s = summarize(scores=[0.9, 0.8, 0.1, 0.2], is_ai=[1, 1, 0, 0], threshold=0.5)
    assert s["accuracy"] == 1.0
    assert s["ai_caught"] == 1.0
    assert s["humans_passed"] == 1.0
    assert s["eer"] == 0.0
    assert s["auc"] == 1.0


def test_everything_called_ai_catches_all_ai_but_fails_all_humans():
    s = summarize(scores=[0.9, 0.9, 0.9, 0.9], is_ai=[1, 1, 0, 0], threshold=0.5)
    assert s["accuracy"] == 0.5
    assert s["ai_caught"] == 1.0
    assert s["humans_passed"] == 0.0


def test_inverted_model_gets_auc_zero_and_eer_one():
    s = summarize(scores=[0.1, 0.2, 0.9, 0.8], is_ai=[1, 1, 0, 0], threshold=0.5)
    assert s["auc"] == 0.0
    assert s["eer"] == 1.0
    assert s["accuracy"] == 0.0


def test_score_exactly_on_threshold_counts_as_ai():
    s = summarize(scores=[0.5, 0.4], is_ai=[1, 0], threshold=0.5)
    assert s["accuracy"] == 1.0


def test_eer_on_overlapping_scores():
    # 4 AI, 4 human; one of each crosses over -> 25% error at the balance point
    ai = [0.9, 0.8, 0.7, 0.3]
    hu = [0.1, 0.2, 0.25, 0.75]
    eer = equal_error_rate(np.array(ai + hu), np.array([1] * 4 + [0] * 4))
    assert eer == pytest.approx(0.25)


def test_counts_are_reported_with_rates():
    s = summarize(scores=[0.9, 0.1, 0.1], is_ai=[1, 1, 0], threshold=0.5)
    assert s["n_ai"] == 2 and s["n_human"] == 1
    assert s["ai_caught"] == 0.5


def test_one_class_only_gives_nan_for_threshold_free_measures():
    s = summarize(scores=[0.9, 0.1], is_ai=[1, 1], threshold=0.5)
    assert math.isnan(s["auc"]) and math.isnan(s["eer"])
    assert s["ai_caught"] == 0.5
    assert math.isnan(s["humans_passed"])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        summarize(scores=[0.1], is_ai=[1, 0], threshold=0.5)
