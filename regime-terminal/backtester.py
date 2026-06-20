"""
backtester.py
-------------
Sequential bar-by-bar backtest engine. Combines:
    - regime signal (from regimelabeler, via HMM states)
    - confirmation count (from strategies.py)
    - risk rules: leverage, cooldown, min-hold, stop-loss, trailing stop,
      immediate exit on regime flip to bear/crash

Entry logic:
    Enter long only if:
        - current regime is bullish (weak_bull / bull / strong_bull)
        - confirmations_count >= cfg.confirmations_required
        - not in cooldown
        - not currently in a position

Exit logic (checked every bar while in a position, in this priority order):
    1. Stop-loss hit
    2. Trailing-stop hit (if enabled)
    3. Regime flips to bearish/crash -> immediate exit (overrides min-hold)
    4. Min-hold satisfied AND confirmations have dropped below threshold

Public API:
    class Trade (dataclass)
    run_backtest(df, regime_names, confirmations_df, cfg) -> BacktestResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import Config
from regimelabeler import is_bullish, is_bearish_or_crash


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    size: float = 1.0          # position size in base-currency units (notional / entry_price)
    leverage: float = 1.0
    exit_reason: str = ""
    pnl: float = 0.0           # realized PnL in quote currency, leveraged, net of costs
    pnl_pct: float = 0.0       # return on margin, net of costs
    equity_after: float = 0.0
    entry_regime: str = ""
    entry_confirmations: int = 0


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = None
    benchmark_curve: pd.Series = None  # buy-and-hold equity for comparison
    metrics: dict = field(default_factory=dict)
    per_state_performance: pd.DataFrame = None


def _make_trades_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_time", "entry_price", "exit_time", "exit_price", "size",
                "leverage", "exit_reason", "pnl", "pnl_pct", "equity_after",
                "entry_regime", "entry_confirmations",
            ]
        )
    return pd.DataFrame([t.__dict__ for t in trades])


def run_backtest(
    df: pd.DataFrame,
    regime_names: np.ndarray,
    confirmations_counts: np.ndarray,
    cfg: Config,
) -> BacktestResult:
    """Run the sequential backtest.

    Parameters
    ----------
    df : OHLCV DataFrame, must be aligned/same length as regime_names & confirmations_counts
    regime_names : array of regime name strings, one per bar (e.g. "bull", "crash", ...)
    confirmations_counts : array of ints, one per bar, count of confirmations passed (bull side)
    cfg : Config

    Returns
    -------
    BacktestResult
    """
    assert len(df) == len(regime_names) == len(confirmations_counts), \
        "df, regime_names, and confirmations_counts must be the same length"

    close = df["Close"].values
    times = df.index

    equity = cfg.initial_equity
    equity_curve = np.empty(len(df))

    in_position = False
    entry_price = 0.0
    entry_time = None
    entry_idx = -1
    entry_regime = ""
    entry_confirmations = 0
    peak_price_since_entry = 0.0
    cooldown_until_idx = -1

    trades: List[Trade] = []

    for i in range(len(df)):
        price = close[i]
        regime = regime_names[i]
        conf_count = confirmations_counts[i]

        if in_position:
            peak_price_since_entry = max(peak_price_since_entry, price)
            hold_hours = i - entry_idx
            exit_reason = None

            # 1. Stop-loss
            drawdown_from_entry = (price - entry_price) / entry_price
            if drawdown_from_entry <= -cfg.stop_loss_pct:
                exit_reason = "stop_loss"

            # 2. Trailing stop
            if exit_reason is None and cfg.use_trailing_stop:
                drawdown_from_peak = (price - peak_price_since_entry) / peak_price_since_entry
                if drawdown_from_peak <= -cfg.trailing_stop_pct:
                    exit_reason = "trailing_stop"

            # 3. Immediate exit on regime flip to bear/crash (overrides min-hold)
            if exit_reason is None and is_bearish_or_crash(regime):
                exit_reason = "regime_flip"

            # 4. Min-hold satisfied AND confirmations dropped below threshold
            if exit_reason is None and hold_hours >= cfg.min_hold_hours:
                if conf_count < cfg.confirmations_required:
                    exit_reason = "confirmations_dropped"

            if exit_reason is not None:
                gross_return = (price - entry_price) / entry_price
                leveraged_return = gross_return * cfg.leverage
                # trade_cost charged on entry AND exit (round trip)
                cost = 2 * cfg.trade_cost * cfg.leverage
                net_return = leveraged_return - cost

                pnl = equity * net_return
                equity = equity + pnl

                trade = Trade(
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=times[i],
                    exit_price=price,
                    size=1.0,
                    leverage=cfg.leverage,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    pnl_pct=net_return,
                    equity_after=equity,
                    entry_regime=entry_regime,
                    entry_confirmations=entry_confirmations,
                )
                trades.append(trade)

                in_position = False
                cooldown_until_idx = i + cfg.cooldown_hours

        else:
            in_cooldown = i < cooldown_until_idx
            if not in_cooldown and is_bullish(regime) and conf_count >= cfg.confirmations_required:
                in_position = True
                entry_price = price
                entry_time = times[i]
                entry_idx = i
                entry_regime = regime
                entry_confirmations = int(conf_count)
                peak_price_since_entry = price

        equity_curve[i] = equity

    # Close any open position at the final bar (mark-to-market)
    if in_position:
        price = close[-1]
        gross_return = (price - entry_price) / entry_price
        leveraged_return = gross_return * cfg.leverage
        cost = 2 * cfg.trade_cost * cfg.leverage
        net_return = leveraged_return - cost
        pnl = equity * net_return
        equity = equity + pnl
        trades.append(
            Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=times[-1],
                exit_price=price,
                size=1.0,
                leverage=cfg.leverage,
                exit_reason="end_of_backtest",
                pnl=pnl,
                pnl_pct=net_return,
                equity_after=equity,
                entry_regime=entry_regime,
                entry_confirmations=entry_confirmations,
            )
        )
        equity_curve[-1] = equity

    equity_series = pd.Series(equity_curve, index=times, name="equity")

    # Buy-and-hold benchmark for comparison
    benchmark = cfg.initial_equity * (df["Close"] / df["Close"].iloc[0])
    benchmark.name = "benchmark"

    trades_df = _make_trades_df(trades)
    metrics = _compute_metrics(equity_series, benchmark, trades_df, cfg)
    per_state_perf = _per_state_performance(trades_df)

    result = BacktestResult(
        trades=trades,
        equity_curve=equity_series,
        benchmark_curve=benchmark,
        metrics=metrics,
        per_state_performance=per_state_perf,
    )
    return result


def _per_state_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["entry_regime", "n_trades", "win_rate", "avg_pnl_pct", "total_pnl"])
    grouped = trades_df.groupby("entry_regime").agg(
        n_trades=("pnl", "count"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        avg_pnl_pct=("pnl_pct", "mean"),
        total_pnl=("pnl", "sum"),
    )
    return grouped.sort_values("total_pnl", ascending=False)


def _compute_metrics(equity: pd.Series, benchmark: pd.Series, trades_df: pd.DataFrame, cfg: Config) -> dict:
    from utils import sharpe_ratio, sortino_ratio, max_drawdown

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    benchmark_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    alpha = total_return - benchmark_return

    hourly_returns = equity.pct_change().dropna()

    n_trades = len(trades_df)
    win_rate = (trades_df["pnl"] > 0).mean() if n_trades > 0 else 0.0
    avg_trade_pnl_pct = trades_df["pnl_pct"].mean() if n_trades > 0 else 0.0

    return {
        "total_return_pct": total_return * 100,
        "benchmark_return_pct": benchmark_return * 100,
        "alpha_pct": alpha * 100,
        "sharpe": sharpe_ratio(hourly_returns, periods_per_year=24 * 365),
        "sortino": sortino_ratio(hourly_returns, periods_per_year=24 * 365),
        "max_drawdown_pct": max_drawdown(equity) * 100,
        "n_trades": n_trades,
        "win_rate_pct": win_rate * 100,
        "avg_trade_pnl_pct": avg_trade_pnl_pct * 100,
        "final_equity": equity.iloc[-1],
    }


if __name__ == "__main__":
    import pandas as pd
    from features import compute_features, scale_features
    from hmmmodel import fit_regime_model
    from regimelabeler import label_regimes, apply_labels
    from strategies import compute_indicators, evaluate_confirmations, confirmations_count

    cfg = Config()
    df = pd.read_parquet(".cache/synthetic_test_data.parquet")

    feats = compute_features(df, cfg)
    scaled, scaler = scale_features(feats)
    model = fit_regime_model(scaled, cfg)
    states = model.predict_states(scaled)
    summary, mapping = label_regimes(feats["returns"], states, cfg)
    regime_names = apply_labels(states, mapping)

    aligned_df = df.loc[feats.index]
    indicators = compute_indicators(aligned_df, cfg).dropna()

    common_idx = indicators.index.intersection(feats.index)
    aligned_df = aligned_df.loc[common_idx]
    regime_names_aligned = pd.Series(regime_names, index=feats.index).loc[common_idx].values
    indicators_aligned = indicators.loc[common_idx]

    conf_counts = []
    for _, row in indicators_aligned.iterrows():
        c = evaluate_confirmations(row, "bull", cfg)
        conf_counts.append(confirmations_count(c))
    conf_counts = np.array(conf_counts)

    result = run_backtest(aligned_df, regime_names_aligned, conf_counts, cfg)
    print("Metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nPer-state performance:")
    print(result.per_state_performance)

    print(f"\nTotal trades: {len(result.trades)}")
    if result.trades:
        print("First trade:", result.trades[0])
