import numpy as np
import pandas as pd
import pytest

from vad_bench.combo import (COMBOS, bootstrap_ci, fit_combo, grouped_folds, pella_prob, veri_prob)


def test_probabilities_follow_each_models_documented_calibration():
    assert pella_prob(np.array([0.0]))[0] == pytest.approx(0.5)
    # VeriSpoof card: P(spoof) = sigmoid(0.930*score - 0.872) -> 50% at score 0.9376
    assert veri_prob(np.array([0.872 / 0.930]))[0] == pytest.approx(0.5)
    assert veri_prob(np.array([10.0]))[0] > 0.99


def toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    # Pella catches only half the AI; Veri catches the other half -> combos should beat either
    pella = np.where(y == 1, np.where(np.arange(n) % 4 == 1, 6.0, -6.0), -7.0) + rng.normal(0, 0.1, n)
    veri = np.where(y == 1, np.where(np.arange(n) % 4 == 3, 5.0, -5.0), -6.0) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"pella": pella, "veri": veri, "is_ai": y, "voice_id": np.arange(n) // 2})


@pytest.mark.parametrize("name", list(COMBOS))
def test_every_combo_fits_and_scores_higher_is_more_ai(name):
    d = toy()
    model = fit_combo(name, d)
    s = model(d)
    assert s.shape == (len(d),)
    assert np.mean(s[d.is_ai == 1]) > np.mean(s[d.is_ai == 0])


def test_either_catches_what_each_model_misses():
    d = toy()
    s = fit_combo("either flags", d)(d)
    cut = 0.5
    assert np.mean(s[d.is_ai == 1] >= cut) == 1.0
    assert np.mean(s[d.is_ai == 0] >= cut) == 0.0


def test_both_must_flag_is_strict():
    d = toy()
    s = fit_combo("both must flag", d)(d)
    assert np.mean(s[d.is_ai == 1] >= 0.5) == 0.0  # never both on this toy data


def test_single_models_are_available_as_combos_too():
    d = toy()
    np.testing.assert_allclose(fit_combo("Pellav2 alone", d)(d), d.pella)
    np.testing.assert_allclose(fit_combo("VeriSpoof-Plus alone", d)(d), d.veri)


def test_grouped_folds_never_split_a_voice():
    voices = np.repeat(np.arange(30), 4)
    folds = grouped_folds(voices, k=5, seed=0)
    assert len(folds) == 5
    seen = np.zeros(len(voices), int)
    for train, test in folds:
        assert not set(voices[train]) & set(voices[test])
        seen[test] += 1
    assert np.all(seen == 1)


def test_bootstrap_ci_brackets_the_estimate_and_resamples_voices():
    d = toy(200)
    stat = lambda x: float(np.mean(x.is_ai))
    lo, hi = bootstrap_ci(d, stat, group="voice_id", n=300, seed=1)
    assert lo <= 0.5 <= hi
    assert lo == hi == 0.5  # every voice holds one AI + one human clip, so the share never moves


def test_operating_points_pick_on_train_and_grade_on_test():
    from vad_bench.combo import operating_points

    tr_s = np.r_[np.arange(100) / 100, [5.0] * 20]  # humans 0..0.99, AI at 5
    tr_y = np.r_[np.zeros(100), np.ones(20)]
    te_s = np.r_[np.full(50, 0.5), np.full(50, 3.0)]
    te_y = np.r_[np.zeros(50), np.ones(50)]
    ops = operating_points(tr_s, tr_y, te_s, te_y, built_in=0.5).set_index("cutoff_kind")
    # strict 5%: cut-off just above the 95th human score (~0.95) -> test humans at 0.5 all pass
    assert ops.loc["≤5% humans flagged"].humans_passed == 1.0
    assert ops.loc["≤5% humans flagged"].ai_caught == 1.0
    assert ops.loc["built-in"].cutoff == 0.5
    assert set(ops.index) == {"built-in", "≤5% humans flagged", "≤1% humans flagged"}


def test_cv_predictions_cover_every_clip_once_and_respect_voices():
    from vad_bench.combo import cv_predictions

    d = toy(60)
    out = cv_predictions(d, "learned weighting", k=5, seed=0)
    assert len(out) == len(d) and out.index.is_unique
    assert {"score", "cutoff_5", "cutoff_1", "fold"} <= set(out.columns)
    for f, part in out.groupby("fold"):
        assert part.voice_id.nunique() == len(set(part.voice_id))  # trivially true; real check below
    folds = out.groupby("voice_id").fold.nunique()
    assert (folds == 1).all()
