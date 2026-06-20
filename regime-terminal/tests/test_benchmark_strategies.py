"""
Unit tests for benchmark_strategies.py

Uses hand-constructed price sequences with known crossover points so we
can assert exact signal behavior, the same approach used in
tests/test_backtester.py for the main strategy.
"""

import numpy as np
import pandas as pd
import pytest

from config import Config
from benchmark_strategies import (
    moving_average_crossover_signal,
    rsi_mean_reversion_signal,
    macd_crossover_signal,
    buy_and_hold_signal,
    run_simple_backtest,
    compare_strategies,
    STANDARD_STRATEGIES,
)


def _make_df(prices, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="h")
    prices = np.array(prices, dtype=float)
    return pd.DataFrame(
        {"Open": prices, "High": prices * 1.001, "Low": prices * 0.999,
         "Close": prices, "Volume": np.full(len(prices), 1000.0)},
        index=idx,
    )


def test_buy_and_hold_signal_always_true():
    df = _make_df([100, 101, 99, 102, 98])
    signal = buy_and_hold_signal(df)
    assert signal.all()
    assert len(signal) == len(df)


def test_moving_average_crossover_signal_basic_uptrend():
    # A clear, sustained uptrend should eventually have fast MA > slow MA
    n = 250
    prices = np.linspace(100, 200, n)  # smooth uptrend
    df = _make_df(prices)
    signal = moving_average_crossover_signal(df, fast=10, slow=50)
    # Once both MAs have enough data and the trend is established, signal should be True
    assert signal[-1] == True  # noqa: E712


def test_moving_average_crossover_signal_false_before_enough_data():
    n = 30  # fewer bars than the slow MA window
    prices = np.linspace(100, 110, n)
    df = _make_df(prices)
    signal = moving_average_crossover_signal(df, fast=10, slow=50)
    # slow MA can't be computed yet for any bar -> should never be True
    assert not signal.any()


def test_moving_average_crossover_signal_downtrend_is_false():
    n = 250
    prices = np.linspace(200, 100, n)  # smooth downtrend
    df = _make_df(prices)
    signal = moving_average_crossover_signal(df, fast=10, slow=50)
    assert signal[-1] == False  # noqa: E712


def test_rsi_mean_reversion_enters_on_oversold_exits_on_overbought():
    # Construct a sequence that drops sharply (RSI low) then rallies sharply (RSI high)
    n = 60
    drop = np.linspace(100, 70, 30)
    rally = np.linspace(70, 130, 30)
    prices = np.concatenate([drop, rally])
    df = _make_df(prices)
    signal = rsi_mean_reversion_signal(df, period=14, oversold=30, overbought=70)

    # Should have entered at some point during/after the drop and exited during the rally
    assert signal.any()
    assert not signal.all()  # shouldn't be in position for the entire series


def test_rsi_mean_reversion_never_enters_in_flat_market():
    n = 60
    prices = np.full(n, 100.0)  # perfectly flat -> RSI stays neutral (~50)
    df = _make_df(prices)
    signal = rsi_mean_reversion_signal(df, period=14, oversold=30, overbought=70)
    assert not signal.any()


def test_macd_crossover_signal_matches_macd_line_vs_signal_line():
    from strategies import _macd
    n = 100
    prices = 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, n))  # oscillating series
    df = _make_df(prices)
    signal = macd_crossover_signal(df, fast=12, slow=26, signal=9)

    macd_line, signal_line = _macd(df["Close"], 12, 26, 9)
    expected = (macd_line > signal_line).fillna(False).values
    np.testing.assert_array_equal(signal, expected)


def test_run_simple_backtest_enters_and_exits_on_signal():
    n = 10
    df = _make_df([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    signal = np.array([False, False, True, True, True, False, False, False, False, False])
    cfg = Config(leverage=1.0, trade_cost=0.0, stop_loss_pct=0.5, use_trailing_stop=False)

    result = run_simple_backtest(df, signal, cfg)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "signal_exit"
    entry_idx = df.index.get_loc(trade.entry_time)
    exit_idx = df.index.get_loc(trade.exit_time)
    assert entry_idx == 2  # signal turned True at index 2
    assert exit_idx == 5   # signal turned False at index 5


def test_run_simple_backtest_stop_loss_triggers_before_signal_exit():
    n = 10
    df = _make_df([100, 100, 70, 69, 68, 67, 66, 65, 64, 63])  # crashes hard after entry
    signal = np.array([False, True, True, True, True, True, True, True, True, True])
    cfg = Config(leverage=1.0, trade_cost=0.0, stop_loss_pct=0.10, use_trailing_stop=False)

    result = run_simple_backtest(df, signal, cfg)
    assert len(result.trades) >= 1
    assert result.trades[0].exit_reason == "stop_loss"


def test_run_simple_backtest_no_signal_produces_no_trades():
    n = 10
    df = _make_df([100] * n)
    signal = np.zeros(n, dtype=bool)
    cfg = Config()

    result = run_simple_backtest(df, signal, cfg)
    assert len(result.trades) == 0
    assert result.metrics["n_trades"] == 0


def test_run_simple_backtest_requires_matching_lengths():
    df = _make_df([100, 101, 102])
    signal = np.array([True, False])  # wrong length
    cfg = Config()
    with pytest.raises(AssertionError):
        run_simple_backtest(df, signal, cfg)


def test_compare_strategies_returns_one_row_per_strategy():
    n = 300
    prices = 100 + np.cumsum(np.random.RandomState(0).normal(0, 0.5, n))
    prices = np.clip(prices, 50, None)  # keep prices positive
    df = _make_df(prices)
    cfg = Config()

    comparison = compare_strategies(df, cfg)
    assert len(comparison) == len(STANDARD_STRATEGIES)
    assert set(comparison.index) == set(STANDARD_STRATEGIES.keys())


def test_compare_strategies_includes_expected_metric_columns():
    n = 300
    prices = 100 + np.cumsum(np.random.RandomState(1).normal(0, 0.5, n))
    prices = np.clip(prices, 50, None)
    df = _make_df(prices)
    cfg = Config()

    comparison = compare_strategies(df, cfg)
    expected_cols = {
        "total_return_pct", "benchmark_return_pct", "alpha_pct",
        "sharpe", "sortino", "max_drawdown_pct", "n_trades", "win_rate_pct",
    }
    assert expected_cols.issubset(set(comparison.columns))


def test_compare_strategies_can_run_subset_of_strategies():
    n = 300
    prices = 100 + np.cumsum(np.random.RandomState(2).normal(0, 0.5, n))
    prices = np.clip(prices, 50, None)
    df = _make_df(prices)
    cfg = Config()

    subset = {"buy_and_hold": buy_and_hold_signal}
    comparison = compare_strategies(df, cfg, strategies=subset)
    assert len(comparison) == 1
    assert comparison.index[0] == "buy_and_hold"


def test_buy_and_hold_in_compare_strategies_matches_total_return_pct_of_benchmark():
    """Sanity check: buy_and_hold's own total_return_pct should be very close
    to its benchmark_return_pct (both are buy-and-hold, modulo leverage/cost
    settings), confirming the comparison table isn't silently miscomputing."""
    n = 200
    prices = 100 + np.cumsum(np.random.RandomState(3).normal(0, 0.3, n))
    prices = np.clip(prices, 50, None)
    df = _make_df(prices)
    cfg = Config(leverage=1.0, trade_cost=0.0)

    comparison = compare_strategies(df, cfg, strategies={"buy_and_hold": buy_and_hold_signal})
    row = comparison.loc["buy_and_hold"]
    assert abs(row["total_return_pct"] - row["benchmark_return_pct"]) < 1.0
