"""Unit tests for strategies.py"""

import numpy as np
import pandas as pd

from config import Config
from strategies import (
    compute_indicators,
    evaluate_confirmations,
    confirmations_count,
    CONFIRMATION_NAMES,
)


def test_compute_indicators_returns_expected_columns(synthetic_ohlcv, small_config):
    indicators = compute_indicators(synthetic_ohlcv, small_config)
    expected_cols = {
        "rsi", "roc", "adx", "macd_line", "macd_signal",
        "realized_vol", "realized_vol_pct", "volume_z",
        "breakout_up", "breakout_down", "ma", "close",
    }
    assert expected_cols.issubset(set(indicators.columns))


def test_rsi_bounded_0_100(synthetic_ohlcv, small_config):
    indicators = compute_indicators(synthetic_ohlcv, small_config)
    rsi = indicators["rsi"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_evaluate_confirmations_returns_all_8_keys(synthetic_ohlcv, small_config):
    indicators = compute_indicators(synthetic_ohlcv, small_config).dropna()
    row = indicators.iloc[-1]
    confirmations = evaluate_confirmations(row, "bull", small_config)
    assert set(confirmations.keys()) == set(CONFIRMATION_NAMES)
    assert len(confirmations) == 8


def test_confirmations_count_matches_true_values(synthetic_ohlcv, small_config):
    indicators = compute_indicators(synthetic_ohlcv, small_config).dropna()
    row = indicators.iloc[-1]
    confirmations = evaluate_confirmations(row, "bull", small_config)
    expected = sum(1 for v in confirmations.values() if v)
    assert confirmations_count(confirmations) == expected


def test_bull_and_bear_confirmations_are_not_simultaneously_all_true():
    """Sanity check: on a clear strong-uptrend synthetic series, the bull-direction
    RSI/momentum/MACD/price-action/MA confirmations should fire more often than
    the equivalent bear-direction confirmations would for the same data."""
    rng = np.random.RandomState(0)
    n = 300
    returns = rng.normal(0.002, 0.003, n)  # clear uptrend
    price = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "Open": price, "High": price * 1.001, "Low": price * 0.999,
            "Close": price, "Volume": rng.lognormal(8, 0.2, n),
        },
        index=idx,
    )
    cfg = Config(ma_period=20)
    indicators = compute_indicators(df, cfg).dropna()
    row = indicators.iloc[-1]

    bull_conf = evaluate_confirmations(row, "bull", cfg)
    bear_conf = evaluate_confirmations(row, "bear", cfg)

    bull_count = confirmations_count(bull_conf)
    bear_count = confirmations_count(bear_conf)
    assert bull_count > bear_count


def test_direction_agnostic_confirmations_identical_across_directions(synthetic_ohlcv, small_config):
    indicators = compute_indicators(synthetic_ohlcv, small_config).dropna()
    row = indicators.iloc[-1]
    bull_conf = evaluate_confirmations(row, "bull", small_config)
    bear_conf = evaluate_confirmations(row, "bear", small_config)
    for key in ["adx", "volatility", "volume_spike"]:
        assert bull_conf[key] == bear_conf[key]


def test_confirmations_count_series_matches_row_by_row(synthetic_ohlcv, small_config):
    """The vectorized confirmations_count_series() must produce exactly the same
    counts as calling evaluate_confirmations()+confirmations_count() per row —
    it's a performance optimization, not a behavior change."""
    from strategies import confirmations_count_series

    indicators = compute_indicators(synthetic_ohlcv, small_config).dropna()

    expected = np.array([
        confirmations_count(evaluate_confirmations(row, "bull", small_config))
        for _, row in indicators.iterrows()
    ])
    actual = confirmations_count_series(indicators, "bull", small_config)

    np.testing.assert_array_equal(actual, expected)


def test_confirmations_count_series_bear_direction_matches_row_by_row(synthetic_ohlcv, small_config):
    from strategies import confirmations_count_series

    indicators = compute_indicators(synthetic_ohlcv, small_config).dropna()

    expected = np.array([
        confirmations_count(evaluate_confirmations(row, "bear", small_config))
        for _, row in indicators.iterrows()
    ])
    actual = confirmations_count_series(indicators, "bear", small_config)

    np.testing.assert_array_equal(actual, expected)
