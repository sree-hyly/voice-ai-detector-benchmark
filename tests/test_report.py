import pandas as pd
import pytest

from vad_bench.report import COMPARISONS, per_source_catch_rate, results_table


def fake_scores():
    rows = []
    # model A: perfect on studio AI, misses all phone AI
    for i in range(4):
        rows.append(("A", f"h{i}", "real_phone", "callhome", 0, -3.0))
        rows.append(("A", f"c{i}", "ai_clean", "el" if i < 2 else "edge", 1, 3.0))
        rows.append(("A", f"p{i}", "ai_phone", "el (via phone line)", 1, -1.0))
    # model B: flags everything as AI
    for i in range(4):
        rows.append(("B", f"h{i}", "real_phone", "callhome", 0, 2.0))
        rows.append(("B", f"c{i}", "ai_clean", "el", 1, 2.0))
        rows.append(("B", f"p{i}", "ai_phone", "el (via phone line)", 1, 2.0))
    return pd.DataFrame(rows, columns=["model", "file", "group", "source", "is_ai", "ai_score"])


def test_one_row_per_model_and_comparison():
    t = results_table(fake_scores())
    assert len(t) == 2 * len(COMPARISONS)
    assert set(t.comparison) == set(COMPARISONS)


def test_studio_ai_vs_phone_humans_uses_only_those_groups():
    t = results_table(fake_scores()).set_index(["model", "comparison"])
    a = t.loc[("A", "studio AI vs real phone")]
    assert a.n_ai == 4 and a.n_human == 4
    assert a.accuracy == 1.0 and a.auc == 1.0


def test_phone_ai_comparison_exposes_the_miss():
    t = results_table(fake_scores()).set_index(["model", "comparison"])
    a = t.loc[("A", "phone AI vs real phone")]
    assert a.ai_caught == 0.0 and a.humans_passed == 1.0 and a.accuracy == 0.5
    # threshold-free view: phone-AI (-1) still scores above humans (-3)
    assert a.auc == 1.0


def test_all_clips_comparison_pools_both_ai_groups():
    t = results_table(fake_scores()).set_index(["model", "comparison"])
    assert t.loc[("A", "all clips")].n_ai == 8


def test_flag_everything_model_passes_no_humans():
    t = results_table(fake_scores()).set_index(["model", "comparison"])
    b = t.loc[("B", "all clips")]
    assert b.ai_caught == 1.0 and b.humans_passed == 0.0 and b.auc == pytest.approx(0.5)


def test_per_source_catch_rate():
    c = per_source_catch_rate(fake_scores())
    a = c.loc["A"]
    assert a["el"] == 1.0 and a["edge"] == 1.0 and a["el (via phone line)"] == 0.0
    assert "callhome" not in c.columns  # humans are not an AI source


def two_sets():
    rows = []
    # model M separates around -5 in both sets (built-in 0 calls everything human)
    for i in range(5):
        rows += [("a", "M", 0, -9.0 + i * 0.1), ("a", "M", 1, -2.0 + i * 0.1),
                 ("b", "M", 0, -8.0 + i * 0.1), ("b", "M", 1, -1.0 + i * 0.1)]
    df = pd.DataFrame(rows, columns=["set", "model", "is_ai", "ai_score"])
    df["group"] = df.is_ai.map({0: "h", 1: "ai"})
    return df[df.set == "a"].drop(columns="set"), df[df.set == "b"].drop(columns="set")


def test_cross_set_grades_on_the_other_set():
    from vad_bench.report import cross_set_table

    a, b = two_sets()
    t = cross_set_table(a, b, set_names=("set 1", "set 2")).set_index(["model", "picked_on", "cutoff_kind"])
    tuned = t.loc[("M", "set 1", "balanced")]
    assert tuned.graded_on == "set 2"
    assert -8 < tuned.cutoff < -2  # learned on set 1 only
    assert tuned.ai_caught == 1.0 and tuned.humans_passed == 1.0


def test_cross_set_includes_built_in_cutoff_for_comparison():
    from vad_bench.report import cross_set_table

    a, b = two_sets()
    t = cross_set_table(a, b, set_names=("set 1", "set 2"))
    built = t[(t.cutoff_kind == "built-in") & (t.graded_on == "set 2")].iloc[0]
    assert built.cutoff == 0.0 and built.ai_caught == 0.0 and built.humans_passed == 1.0


def test_cross_set_has_both_directions_and_three_kinds():
    from vad_bench.report import cross_set_table

    a, b = two_sets()
    t = cross_set_table(a, b, set_names=("set 1", "set 2"))
    assert set(t.graded_on) == {"set 1", "set 2"}
    assert set(t.cutoff_kind) == {"built-in", "balanced", "strict (≤5% humans flagged)"}
