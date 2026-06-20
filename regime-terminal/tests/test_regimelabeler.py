"""Unit tests for regimelabeler.py"""

import numpy as np
import pandas as pd
import pytest

from config import Config
from regimelabeler import label_regimes, apply_labels, is_bullish, is_bearish_or_crash


def test_label_regimes_highest_return_state_is_most_bullish():
    # 3 synthetic states with clearly different mean returns
    returns = pd.Series(
        [0.01] * 100 + [0.0] * 100 + [-0.01] * 100
    )
    states = np.array([5] * 100 + [2] * 100 + [9] * 100)  # arbitrary non-sequential ids
    cfg = Config(n_components=3)

    summary, mapping = label_regimes(returns, states, cfg)

    assert mapping[5] in ("bull", "strong_bull", "weak_bull")
    assert mapping[9] in ("crash", "bear", "weak_bear")


def test_label_regimes_summary_has_expected_columns():
    returns = pd.Series(np.random.RandomState(0).normal(0, 0.01, 300))
    states = np.random.RandomState(0).choice([0, 1, 2], size=300)
    cfg = Config(n_components=3)
    summary, mapping = label_regimes(returns, states, cfg)
    for col in ["mean_return", "volatility", "count", "pct_of_time", "regime_name"]:
        assert col in summary.columns


def test_apply_labels_maps_correctly():
    states = np.array([0, 1, 2, 1, 0])
    mapping = {0: "bull", 1: "chop", 2: "crash"}
    labels = apply_labels(states, mapping)
    assert list(labels) == ["bull", "chop", "crash", "chop", "bull"]


def test_is_bullish_and_bearish_classification():
    assert is_bullish("bull")
    assert is_bullish("strong_bull")
    assert is_bullish("weak_bull")
    assert not is_bullish("chop")
    assert not is_bullish("crash")

    assert is_bearish_or_crash("crash")
    assert is_bearish_or_crash("bear")
    assert not is_bearish_or_crash("bull")
    assert not is_bearish_or_crash("chop")


def test_mapping_covers_all_observed_states():
    returns = pd.Series(np.random.RandomState(1).normal(0, 0.01, 500))
    states = np.random.RandomState(1).choice([3, 7, 1], size=500)
    cfg = Config(n_components=3)
    summary, mapping = label_regimes(returns, states, cfg)
    assert set(mapping.keys()) == {1, 3, 7}


def test_compute_transition_table_rows_sum_to_one():
    from regimelabeler import compute_transition_table
    # hand-built sequence with known transitions
    regimes = np.array(["bull", "bull", "chop", "bull", "chop", "crash", "bull"])
    table = compute_transition_table(regimes)
    pct_cols = [c for c in table.columns if c != "n_observations"]
    row_sums = table[pct_cols].sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-9)


def test_compute_transition_table_known_sequence_exact_values():
    from regimelabeler import compute_transition_table
    # "bull" is followed by: bull, chop, chop -> out of 3 times after bull,
    # 1 time it's bull again, 2 times it's chop
    regimes = np.array(["bull", "bull", "chop", "bull", "chop", "crash"])
    table = compute_transition_table(regimes)
    # bull occurs at indices 0,1,3 (not counting the last element, since there's
    # no "next" after the final bar) -> next values are bull, chop, chop
    assert table.loc["bull", "n_observations"] == 3
    assert table.loc["bull", "bull"] == pytest.approx(1 / 3)
    assert table.loc["bull", "chop"] == pytest.approx(2 / 3)


def test_compute_transition_table_empty_for_short_sequence():
    from regimelabeler import compute_transition_table
    assert compute_transition_table(np.array(["bull"])).empty
    assert compute_transition_table(np.array([])).empty


def test_most_likely_next_regimes_returns_sorted_top_n():
    from regimelabeler import compute_transition_table, most_likely_next_regimes
    regimes = np.array(["bull"] * 10 + ["chop"] * 3 + ["bull"] + ["crash"])
    table = compute_transition_table(regimes)
    top = most_likely_next_regimes(table, "bull", top_n=2)
    assert len(top) == 2
    # results should be sorted descending by probability
    assert top[0][1] >= top[1][1]


def test_most_likely_next_regimes_unknown_regime_returns_empty():
    from regimelabeler import compute_transition_table, most_likely_next_regimes
    regimes = np.array(["bull", "chop", "bull"])
    table = compute_transition_table(regimes)
    assert most_likely_next_regimes(table, "nonexistent_regime") == []
