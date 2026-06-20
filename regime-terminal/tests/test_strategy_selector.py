"""
Unit tests for strategy_selector.py

Uses hand-constructed scenarios with known winning strategies per regime,
so we can assert the table picks the genuinely best strategy and that the
walk-forward wrapper never lets the lookup table see test-window data.
"""

import numpy as np
import pandas as pd
import pytest

from config import Config
from strategy_selector import (
    build_regime_strategy_table,
    apply_strategy_table,
    run_selector_walkforward,
    DEFAULT_FALLBACK_STRATEGY,
    ALL_CANDIDATE_NAMES,
)


def _make_df(prices, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="h")
    prices = np.array(prices, dtype=float)
    return pd.DataFrame(
        {"Open": prices, "High": prices * 1.001, "Low": prices * 0.999,
         "Close": prices, "Volume": np.full(len(prices), 1000.0)},
        index=idx,
    )


def test_build_regime_strategy_table_falls_back_below_min_observations():
    n = 20  # deliberately small -- below any reasonable min_observations_per_regime
    df = _make_df(np.linspace(100, 110, n))
    regime_names = np.array(["bull"] * n)
    conf_counts = np.full(n, 8)
    cfg = Config(confirmations_required=7)

    table = build_regime_strategy_table(df, regime_names, conf_counts, cfg, min_observations_per_regime=30)
    assert table.loc["bull", "best_strategy"] == DEFAULT_FALLBACK_STRATEGY


def test_build_regime_strategy_table_picks_hmm_strategy_when_it_avoids_a_crash():
    """In a regime where price crashes hard, a strategy that's never in
    position (the HMM strategy during 'crash', by construction) should
    score better than always-long strategies that ride the crash down."""
    n = 100
    prices = np.linspace(100, 50, n)  # straight-line 50% crash
    df = _make_df(prices)
    regime_names = np.array(["crash"] * n)
    conf_counts = np.full(n, 0)  # low confirmations -> HMM strategy stays flat in crash regardless
    cfg = Config(confirmations_required=7)

    table = build_regime_strategy_table(df, regime_names, conf_counts, cfg, min_observations_per_regime=10)
    assert table.loc["crash", "best_strategy"] == "hmm_regime_strategy"
    assert table.loc["crash", "hmm_regime_strategy"] == 0.0  # flat the whole time -> zero contribution
    assert table.loc["crash", "buy_and_hold"] < 0  # lost money riding the crash down


def test_build_regime_strategy_table_picks_buy_and_hold_in_clear_uptrend():
    n = 300
    prices = np.linspace(100, 200, n)  # smooth, sustained uptrend
    df = _make_df(prices)
    regime_names = np.array(["bull"] * n)
    conf_counts = np.full(n, 8)
    cfg = Config(confirmations_required=7)

    table = build_regime_strategy_table(df, regime_names, conf_counts, cfg, min_observations_per_regime=10)
    # Buy-and-hold should win (or tie closely with the HMM strategy, which
    # is also long throughout given high confirmations) in a clean uptrend
    assert table.loc["bull", "best_strategy"] in ("buy_and_hold", "hmm_regime_strategy")


def test_apply_strategy_table_uses_correct_strategy_per_regime():
    n = 20
    df = _make_df(np.linspace(100, 90, n))  # downtrend
    regime_names = np.array(["bear"] * 10 + ["bull"] * 10)
    conf_counts = np.full(n, 8)
    cfg = Config(confirmations_required=7)

    # Hand-construct a table: bear -> always flat (buy_and_hold won't apply since
    # we force it), bull -> always in position via hmm_regime_strategy
    table = pd.DataFrame({
        "best_strategy": ["buy_and_hold", "hmm_regime_strategy"],
        "n_observations": [10, 10],
    }, index=["bear", "bull"])

    signal = apply_strategy_table(df, regime_names, conf_counts, table, cfg)
    # buy_and_hold is always True regardless of regime
    assert signal[:10].all()
    # hmm_regime_strategy with conf_counts=8 >= required=7 and regime="bull" -> True
    assert signal[10:].all()


def test_apply_strategy_table_falls_back_for_unseen_regime():
    n = 10
    df = _make_df(np.linspace(100, 110, n))
    regime_names = np.array(["chop"] * n)  # not present in the table at all
    conf_counts = np.full(n, 8)
    cfg = Config(confirmations_required=7)

    table = pd.DataFrame({
        "best_strategy": ["buy_and_hold"],
        "n_observations": [50],
    }, index=["bull"])  # "chop" is absent

    signal = apply_strategy_table(df, regime_names, conf_counts, table, cfg)
    # Falls back to DEFAULT_FALLBACK_STRATEGY, which is buy_and_hold -> always True
    assert signal.all()


def test_all_candidate_names_includes_hmm_and_all_signal_strategies():
    from strategy_selector import SIGNAL_STRATEGIES
    assert "hmm_regime_strategy" in ALL_CANDIDATE_NAMES
    for name in SIGNAL_STRATEGIES:
        assert name in ALL_CANDIDATE_NAMES


@pytest.fixture
def long_synthetic_ohlcv():
    rng = np.random.RandomState(11)
    n = 3000
    segments = []
    for _ in range(6):
        drift = rng.choice([-0.0008, 0.0, 0.0008])
        vol = rng.choice([0.003, 0.006, 0.01])
        length = n // 6
        segments.append(rng.normal(drift, vol, length))
    returns = np.concatenate(segments)
    price = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=len(price), freq="h")
    high = price * (1 + np.abs(rng.normal(0, 0.001, len(price))))
    low = price * (1 - np.abs(rng.normal(0, 0.001, len(price))))
    openp = price * (1 + rng.normal(0, 0.0005, len(price)))
    volume = rng.lognormal(8, 0.3, len(price))
    return pd.DataFrame(
        {"Open": openp, "High": high, "Low": low, "Close": price, "Volume": volume},
        index=idx,
    )


def test_run_selector_walkforward_returns_valid_result(long_synthetic_ohlcv):
    cfg = Config(n_components=3, hmm_n_iter=100, confirmations_required=5, random_state=42)
    result = run_selector_walkforward(long_synthetic_ohlcv, cfg, n_folds=3, train_frac=0.7)
    assert len(result.windows) > 0
    assert isinstance(result.per_fold_metrics, pd.DataFrame)
    assert isinstance(result.aggregated_metrics, dict)


def test_run_selector_walkforward_train_strictly_precedes_test(long_synthetic_ohlcv):
    """The core guarantee: the strategy table for each fold is built only
    from data strictly before that fold's test window."""
    cfg = Config(n_components=3, hmm_n_iter=100, confirmations_required=5, random_state=42)
    result = run_selector_walkforward(long_synthetic_ohlcv, cfg, n_folds=3, train_frac=0.7)
    for w in result.windows:
        assert w.train_end < w.test_start


def test_run_selector_walkforward_raises_with_too_little_data():
    cfg = Config(n_components=3, random_state=42)
    tiny_idx = pd.date_range("2024-01-01", periods=20, freq="h")
    tiny_df = pd.DataFrame(
        {"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 1000},
        index=tiny_idx,
    )
    with pytest.raises(ValueError):
        run_selector_walkforward(tiny_df, cfg, n_folds=10, train_frac=0.7)
