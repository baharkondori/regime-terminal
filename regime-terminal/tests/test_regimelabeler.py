"""Unit tests for regimelabeler.py"""

import numpy as np
import pandas as pd

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
