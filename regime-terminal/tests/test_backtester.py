"""
Unit tests for backtester.py

These use small, hand-constructed synthetic sequences (not the fixture data)
so we can assert exact entry/exit behavior against known ground truth.
"""

import numpy as np
import pandas as pd
import pytest

from config import Config
from backtester import run_backtest


def _make_df(prices, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="h")
    prices = np.array(prices, dtype=float)
    df = pd.DataFrame(
        {
            "Open": prices, "High": prices * 1.001, "Low": prices * 0.999,
            "Close": prices, "Volume": np.full(len(prices), 1000.0),
        },
        index=idx,
    )
    return df


def test_enters_on_bull_regime_with_enough_confirmations():
    n = 10
    df = _make_df([100 + i for i in range(n)])
    regimes = np.array(["chop"] * 3 + ["bull"] * 7)
    confirmations = np.array([8] * n)  # always max confirmations
    cfg = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=0, leverage=1.0, trade_cost=0.0)

    result = run_backtest(df, regimes, confirmations, cfg)
    assert len(result.trades) >= 1
    # Entry should happen at or after index 3 (when regime becomes "bull")
    first_entry_idx = df.index.get_loc(result.trades[0].entry_time)
    assert first_entry_idx >= 3


def test_does_not_enter_without_enough_confirmations():
    n = 10
    df = _make_df([100 + i for i in range(n)])
    regimes = np.array(["bull"] * n)
    confirmations = np.array([2] * n)  # below threshold the whole time
    cfg = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=0)

    result = run_backtest(df, regimes, confirmations, cfg)
    assert len(result.trades) == 0


def test_does_not_enter_during_bearish_regime():
    n = 10
    df = _make_df([100 - i for i in range(n)])
    regimes = np.array(["crash"] * n)
    confirmations = np.array([8] * n)
    cfg = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=0)

    result = run_backtest(df, regimes, confirmations, cfg)
    assert len(result.trades) == 0


def test_immediate_exit_on_regime_flip_to_crash():
    n = 10
    df = _make_df([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    # Enter on bull at idx 0, flip to crash at idx 3 -> should exit at idx 3
    # even though min_hold_hours is large (immediate exit overrides min-hold)
    regimes = np.array(["bull", "bull", "bull", "crash", "crash", "crash", "crash", "crash", "crash", "crash"])
    confirmations = np.array([8] * n)
    cfg = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=50, leverage=1.0, trade_cost=0.0)

    result = run_backtest(df, regimes, confirmations, cfg)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "regime_flip"
    exit_idx = df.index.get_loc(trade.exit_time)
    assert exit_idx == 3  # exits exactly when regime flips, not later


def test_min_hold_prevents_early_exit_on_confirmation_drop():
    n = 10
    df = _make_df([100 + i * 0.1 for i in range(n)])
    regimes = np.array(["bull"] * n)  # stays bullish throughout
    # confirmations drop below threshold at idx 2, but min_hold=5 should
    # prevent exiting until idx >= entry_idx + 5
    confirmations = np.array([8, 8, 1, 1, 1, 1, 1, 1, 1, 1])
    cfg = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=5, leverage=1.0, trade_cost=0.0)

    result = run_backtest(df, regimes, confirmations, cfg)
    assert len(result.trades) == 1
    trade = result.trades[0]
    entry_idx = df.index.get_loc(trade.entry_time)
    exit_idx = df.index.get_loc(trade.exit_time)
    assert (exit_idx - entry_idx) >= 5
    assert trade.exit_reason == "confirmations_dropped"


def test_cooldown_prevents_immediate_reentry():
    n = 20
    df = _make_df([100 + i for i in range(n)])
    # bull/8-confirmations the whole time except a crash blip at idx 5 to force an exit
    regimes = np.array(["bull"] * 5 + ["crash"] + ["bull"] * 14)
    confirmations = np.array([8] * n)
    cfg = Config(confirmations_required=7, cooldown_hours=5, min_hold_hours=0, leverage=1.0, trade_cost=0.0)

    result = run_backtest(df, regimes, confirmations, cfg)
    assert len(result.trades) >= 2
    first_exit_idx = df.index.get_loc(result.trades[0].exit_time)
    second_entry_idx = df.index.get_loc(result.trades[1].entry_time)
    # Re-entry should not happen until at least cooldown_hours after exit
    assert (second_entry_idx - first_exit_idx) >= cfg.cooldown_hours


def test_stop_loss_triggers_exit():
    n = 10
    # Price crashes hard right after entry
    df = _make_df([100, 100, 80, 79, 78, 77, 76, 75, 74, 73])
    regimes = np.array(["bull"] * n)
    confirmations = np.array([8] * n)
    cfg = Config(
        confirmations_required=7, cooldown_hours=100, min_hold_hours=50,
        stop_loss_pct=0.10, leverage=1.0, trade_cost=0.0,
    )

    result = run_backtest(df, regimes, confirmations, cfg)
    # cooldown=100 ensures only the first stop-out is captured in this short window
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop_loss"


def test_trailing_stop_triggers_after_peak_pullback():
    n = 12
    # Price rises then pulls back from the peak
    prices = [100, 105, 110, 115, 120, 118, 115, 112, 110, 108, 106, 104]
    df = _make_df(prices)
    regimes = np.array(["bull"] * n)
    confirmations = np.array([8] * n)
    cfg = Config(
        confirmations_required=7, cooldown_hours=100, min_hold_hours=50,
        use_trailing_stop=True, trailing_stop_pct=0.03,
        stop_loss_pct=0.5,  # disable stop loss from triggering first
        leverage=1.0, trade_cost=0.0,
    )

    result = run_backtest(df, regimes, confirmations, cfg)
    # cooldown=100 ensures only the first trailing-stop exit is captured here
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "trailing_stop"
    # peak was 120 at idx 4; a 3% pullback from 120 is 116.4, crossed at idx 6 (price 115)
    exit_idx = df.index.get_loc(result.trades[0].exit_time)
    assert exit_idx == 6


def test_leverage_amplifies_returns():
    n = 5
    df = _make_df([100, 101, 102, 103, 104])
    regimes = np.array(["bull"] * n)
    confirmations = np.array([8] * n)

    cfg_1x = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=0, leverage=1.0, trade_cost=0.0)
    cfg_3x = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=0, leverage=3.0, trade_cost=0.0)

    result_1x = run_backtest(df, regimes, confirmations, cfg_1x)
    result_3x = run_backtest(df, regimes, confirmations, cfg_3x)

    pnl_pct_1x = result_1x.trades[0].pnl_pct
    pnl_pct_3x = result_3x.trades[0].pnl_pct
    assert pnl_pct_3x == pytest.approx(pnl_pct_1x * 3, rel=1e-6)


def test_trade_costs_reduce_pnl():
    n = 5
    df = _make_df([100, 101, 102, 103, 104])
    regimes = np.array(["bull"] * n)
    confirmations = np.array([8] * n)

    cfg_no_cost = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=0, leverage=1.0, trade_cost=0.0)
    cfg_with_cost = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=0, leverage=1.0, trade_cost=0.01)

    result_no_cost = run_backtest(df, regimes, confirmations, cfg_no_cost)
    result_with_cost = run_backtest(df, regimes, confirmations, cfg_with_cost)

    assert result_with_cost.trades[0].pnl_pct < result_no_cost.trades[0].pnl_pct


def test_metrics_dict_has_expected_keys():
    n = 30
    df = _make_df([100 + i * 0.5 for i in range(n)])
    regimes = np.array(["bull"] * n)
    confirmations = np.array([8] * n)
    cfg = Config(confirmations_required=7, cooldown_hours=0, min_hold_hours=0)

    result = run_backtest(df, regimes, confirmations, cfg)
    expected_keys = {
        "total_return_pct", "benchmark_return_pct", "alpha_pct", "sharpe",
        "sortino", "max_drawdown_pct", "n_trades", "win_rate_pct",
        "avg_trade_pnl_pct", "final_equity",
    }
    assert expected_keys.issubset(set(result.metrics.keys()))


def test_no_trades_handled_gracefully():
    n = 10
    df = _make_df([100] * n)
    regimes = np.array(["crash"] * n)  # never bullish, never enters
    confirmations = np.array([0] * n)
    cfg = Config(confirmations_required=7)

    result = run_backtest(df, regimes, confirmations, cfg)
    assert len(result.trades) == 0
    assert result.metrics["n_trades"] == 0
    assert result.metrics["win_rate_pct"] == 0.0
