"""
benchmark_strategies.py
------------------------
Standard, widely-known trading strategies, used as honest benchmarks
against this project's custom HMM + confirmations strategy (and, later,
the neural-network classifier in nn_regime_model.py).

Important framing: these are "standard" in the sense of being simple,
widely taught, and widely followed -- not in the sense of having a
proven, durable edge. Comparing against them is a reality check: if the
custom HMM strategy can't beat a plain 50/200 moving-average crossover in
honest walk-forward testing, that's important information, not a reason
to assume something is broken. Simple, well-known strategies frequently
match or beat more complex ones once realistic costs are included --
that's a well-documented pattern in this space, not a knock on either
approach.

Each strategy here is a function with signature:
    strategy_fn(df: pd.DataFrame, **params) -> np.ndarray[bool]
returning a boolean "should be long" signal per bar, the same length as
df. This is intentionally a simpler interface than backtester.run_backtest()
(which is wired specifically to the HMM's regime_names/confirmations
vocabulary) -- standard strategies don't need that machinery, only a
position signal, which run_simple_backtest() then turns into trades using
the same leverage/cost/stop-loss conventions as the rest of this project.

Public API:
    moving_average_crossover_signal(df, fast=50, slow=200) -> np.ndarray[bool]
    rsi_mean_reversion_signal(df, period=14, oversold=30, overbought=70) -> np.ndarray[bool]
    macd_crossover_signal(df, fast=12, slow=26, signal=9) -> np.ndarray[bool]
    buy_and_hold_signal(df) -> np.ndarray[bool]
    run_simple_backtest(df, signal, cfg) -> BacktestResult  (reuses backtester.py's Trade/BacktestResult)
    compare_strategies(df, cfg, strategies=None) -> pd.DataFrame
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from config import Config
from backtester import Trade, BacktestResult, _make_trades_df, _compute_metrics
from strategies import _rsi, _macd


def moving_average_crossover_signal(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> np.ndarray:
    """Classic golden-cross / death-cross signal: long whenever the fast
    moving average is above the slow moving average. About as standard a
    trend-following strategy as exists."""
    close = df["Close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = (fast_ma > slow_ma).fillna(False)
    return signal.values


def rsi_mean_reversion_signal(
    df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70
) -> np.ndarray:
    """Classic RSI mean-reversion: go long when RSI drops into oversold
    territory, exit once it climbs back into/through overbought territory.
    This is a stateful signal (depends on prior position), computed as a
    forward-fill between oversold entries and overbought exits."""
    close = df["Close"]
    rsi = _rsi(close, period)

    in_position = np.zeros(len(df), dtype=bool)
    holding = False
    for i in range(len(df)):
        if not holding and rsi.iloc[i] < oversold:
            holding = True
        elif holding and rsi.iloc[i] > overbought:
            holding = False
        in_position[i] = holding
    return in_position


def macd_crossover_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """Classic MACD crossover: long whenever the MACD line is above its
    signal line."""
    close = df["Close"]
    macd_line, signal_line = _macd(close, fast, slow, signal)
    in_position = (macd_line > signal_line).fillna(False)
    return in_position.values


def buy_and_hold_signal(df: pd.DataFrame) -> np.ndarray:
    """Always long, from the first bar to the last. The baseline every
    other strategy should be honestly compared against, including this
    project's own HMM strategy."""
    return np.ones(len(df), dtype=bool)


def run_simple_backtest(df: pd.DataFrame, signal: np.ndarray, cfg: Config) -> BacktestResult:
    """Backtest a simple boolean position signal (no regime/confirmations
    concepts -- just "in position" or not), using the same leverage,
    trade_cost, and stop-loss conventions as backtester.run_backtest(), so
    results are directly comparable.

    Unlike run_backtest(), this has no cooldown/min-hold/confirmations
    logic -- standard strategies are defined purely by their entry/exit
    signal, so adding this project's specific risk-management rules on top
    would no longer be testing the "standard" version of the strategy.
    Stop-loss is still applied, since omitting risk management entirely
    would make the comparison unfair in the other direction (these
    benchmarks would have unlimited downside while the HMM strategy doesn't).
    """
    assert len(df) == len(signal), "df and signal must be the same length"

    close = df["Close"].values
    times = df.index

    equity = cfg.initial_equity
    equity_curve = np.empty(len(df))

    in_position = False
    entry_price = 0.0
    entry_time = None
    peak_price_since_entry = 0.0

    trades = []

    for i in range(len(df)):
        price = close[i]
        want_long = bool(signal[i])

        if in_position:
            peak_price_since_entry = max(peak_price_since_entry, price)
            drawdown_from_entry = (price - entry_price) / entry_price
            exit_reason = None

            if drawdown_from_entry <= -cfg.stop_loss_pct:
                exit_reason = "stop_loss"
            elif cfg.use_trailing_stop:
                drawdown_from_peak = (price - peak_price_since_entry) / peak_price_since_entry
                if drawdown_from_peak <= -cfg.trailing_stop_pct:
                    exit_reason = "trailing_stop"
            elif not want_long:
                exit_reason = "signal_exit"

            if exit_reason is not None:
                gross_return = (price - entry_price) / entry_price
                leveraged_return = gross_return * cfg.leverage
                cost = 2 * cfg.trade_cost * cfg.leverage
                net_return = leveraged_return - cost
                pnl = equity * net_return
                equity += pnl

                trades.append(Trade(
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=times[i], exit_price=price, size=1.0,
                    leverage=cfg.leverage, exit_reason=exit_reason,
                    pnl=pnl, pnl_pct=net_return, equity_after=equity,
                    entry_regime="n/a", entry_confirmations=0,
                ))
                in_position = False
        else:
            if want_long:
                in_position = True
                entry_price = price
                entry_time = times[i]
                peak_price_since_entry = price

        equity_curve[i] = equity

    if in_position:
        price = close[-1]
        gross_return = (price - entry_price) / entry_price
        leveraged_return = gross_return * cfg.leverage
        cost = 2 * cfg.trade_cost * cfg.leverage
        net_return = leveraged_return - cost
        pnl = equity * net_return
        equity += pnl
        trades.append(Trade(
            entry_time=entry_time, entry_price=entry_price,
            exit_time=times[-1], exit_price=price, size=1.0,
            leverage=cfg.leverage, exit_reason="end_of_backtest",
            pnl=pnl, pnl_pct=net_return, equity_after=equity,
            entry_regime="n/a", entry_confirmations=0,
        ))
        equity_curve[-1] = equity

    equity_series = pd.Series(equity_curve, index=times, name="equity")
    benchmark = cfg.initial_equity * (df["Close"] / df["Close"].iloc[0])
    benchmark.name = "benchmark"

    trades_df = _make_trades_df(trades)
    metrics = _compute_metrics(equity_series, benchmark, trades_df, cfg)

    return BacktestResult(
        trades=trades, equity_curve=equity_series,
        benchmark_curve=benchmark, metrics=metrics,
        per_state_performance=pd.DataFrame(),  # not meaningful for non-regime strategies
    )


STANDARD_STRATEGIES: Dict[str, Callable] = {
    "buy_and_hold": buy_and_hold_signal,
    "ma_crossover_50_200": lambda df: moving_average_crossover_signal(df, fast=50, slow=200),
    "rsi_mean_reversion": lambda df: rsi_mean_reversion_signal(df, period=14, oversold=30, overbought=70),
    "macd_crossover": lambda df: macd_crossover_signal(df, fast=12, slow=26, signal=9),
}


def compare_strategies(
    df: pd.DataFrame,
    cfg: Config,
    strategies: Optional[Dict[str, Callable]] = None,
) -> pd.DataFrame:
    """Run every strategy in `strategies` (default: STANDARD_STRATEGIES)
    through run_simple_backtest() and return a comparison table of key
    metrics, one row per strategy.

    This does NOT include this project's own HMM strategy -- that uses a
    different signal vocabulary (regime + confirmations) and runs through
    backtester.run_backtest() or walkforward.run_walkforward() instead. To
    build a true side-by-side table including the HMM strategy, run that
    separately and concatenate the resulting metrics with this function's
    output -- see cli.py's --compare-strategies command for an example.
    """
    strategies = strategies or STANDARD_STRATEGIES
    rows = []
    for name, strategy_fn in strategies.items():
        signal = strategy_fn(df)
        result = run_simple_backtest(df, signal, cfg)
        m = result.metrics
        rows.append({
            "strategy": name,
            "total_return_pct": m["total_return_pct"],
            "benchmark_return_pct": m["benchmark_return_pct"],
            "alpha_pct": m["alpha_pct"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "max_drawdown_pct": m["max_drawdown_pct"],
            "n_trades": m["n_trades"],
            "win_rate_pct": m["win_rate_pct"],
        })
    return pd.DataFrame(rows).set_index("strategy")


if __name__ == "__main__":
    import pandas as pd

    cfg = Config()
    df = pd.read_parquet(".cache/synthetic_test_data.parquet")

    comparison = compare_strategies(df, cfg)
    print("\n" + "=" * 70)
    print("STANDARD STRATEGY COMPARISON (in-sample, for reference)")
    print("=" * 70)
    print(comparison.to_string())
    print(
        "\nNote: these are simple in-sample results for a quick look, not\n"
        "walk-forward validated. Use walkforward.py for an honest\n"
        "out-of-sample comparison before drawing conclusions."
    )
