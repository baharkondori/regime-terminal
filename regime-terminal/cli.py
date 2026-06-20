"""
cli.py
------
Command-line entry point for running a backtest or producing a live signal
without launching the Streamlit dashboard.

Examples
--------
    python cli.py --run-backtest --asset BTC-USD --lookback-days 730 --leverage 2.5 --confirmations 7
    python cli.py --live-signal --asset BTC-USD
    python cli.py --run-backtest --aggressive
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from config import Config, DISCLAIMER
from dataloader import load_ohlcv
from features import compute_features, scale_features
from hmmmodel import fit_regime_model
from regimelabeler import label_regimes, apply_labels, is_bullish
from strategies import compute_indicators, confirmations_count_series
from backtester import run_backtest
from walkforward import run_walkforward
from benchmark_strategies import compare_strategies
from strategy_selector import run_selector_walkforward
from utils import set_seed


def build_config_from_args(args) -> Config:
    cfg = Config(
        asset=args.asset,
        asset_class=args.asset_class,
        lookback_days=args.lookback_days,
        n_components=args.n_components,
        hmm_backend=args.hmm_backend,
        confirmations_required=args.confirmations,
        leverage=args.leverage,
        min_hold_hours=args.min_hold_hours,
        cooldown_hours=args.cooldown_hours,
        use_trailing_stop=args.trailing_stop,
    )
    if args.aggressive:
        cfg = cfg.apply_aggressive()
    return cfg


def run_pipeline(cfg: Config, force_refresh: bool = False):
    set_seed(cfg.random_state)
    df = load_ohlcv(cfg, force_refresh=force_refresh)

    feats = compute_features(df, cfg)
    scaled, scaler = scale_features(feats)

    model = fit_regime_model(scaled, cfg)
    states = model.predict_states(scaled)
    proba = model.predict_proba(scaled)

    summary, mapping = label_regimes(feats["returns"], states, cfg)
    regime_names = apply_labels(states, mapping)

    aligned_df = df.loc[feats.index]
    indicators = compute_indicators(aligned_df, cfg)
    common_idx = indicators.dropna().index.intersection(feats.index)

    aligned_df = aligned_df.loc[common_idx]
    regime_names_aligned = pd.Series(regime_names, index=feats.index).loc[common_idx].values
    proba_aligned = pd.DataFrame(proba, index=feats.index).loc[common_idx].values
    indicators_aligned = indicators.loc[common_idx]

    conf_counts = confirmations_count_series(indicators_aligned, "bull", cfg)

    return {
        "df": aligned_df,
        "regime_names": regime_names_aligned,
        "proba": proba_aligned,
        "state_summary": summary,
        "mapping": mapping,
        "conf_counts": conf_counts,
        "model": model,
        "indicators": indicators_aligned,
        "scaler": scaler,
    }


def cmd_run_backtest(cfg: Config, force_refresh: bool):
    pipeline = run_pipeline(cfg, force_refresh)
    result = run_backtest(pipeline["df"], pipeline["regime_names"], pipeline["conf_counts"], cfg)

    print(f"\n{'=' * 60}")
    print(f"REGIME TERMINAL BACKTEST — {cfg.asset}")
    print(f"{'=' * 60}")
    print(f"Backend: {pipeline['model'].backend_} | States: {cfg.n_components}")
    print(f"Leverage: {cfg.leverage}x | Confirmations required: {cfg.confirmations_required}/8")
    print(f"Min hold: {cfg.min_hold_hours}h | Cooldown: {cfg.cooldown_hours}h\n")

    print("State summary:")
    print(pipeline["state_summary"].to_string())

    print("\nMetrics:")
    for k, v in result.metrics.items():
        print(f"  {k:25s}: {v:,.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    print("\nPer-state performance:")
    print(result.per_state_performance.to_string())

    print(f"\n{DISCLAIMER}\n")
    return result


def cmd_live_signal(cfg: Config, force_refresh: bool):
    pipeline = run_pipeline(cfg, force_refresh)
    last_regime = pipeline["regime_names"][-1]
    last_proba = float(np.max(pipeline["proba"][-1]))
    last_conf = int(pipeline["conf_counts"][-1])
    bullish = is_bullish(last_regime)
    action = (
        "LONG / HOLD" if (bullish and last_conf >= cfg.confirmations_required)
        else "EXIT / FLAT" if not bullish
        else "WATCH (regime ok, confirmations low)"
    )

    print(f"\n{'=' * 60}")
    print(f"LIVE SIGNAL — {cfg.asset} @ {pipeline['df'].index[-1]}")
    print(f"{'=' * 60}")
    print(f"Regime:            {last_regime}")
    print(f"Regime confidence: {last_proba:.1%}")
    print(f"Confirmations:     {last_conf}/{cfg.n_confirmations_total}")
    print(f"Suggested action:  {action}")
    print(f"\n{DISCLAIMER}\n")


def cmd_walkforward(cfg: Config, force_refresh: bool, n_folds: int, train_frac: float):
    set_seed(cfg.random_state)
    df = load_ohlcv(cfg, force_refresh=force_refresh)

    result = run_walkforward(df, cfg, n_folds=n_folds, train_frac=train_frac)

    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD VALIDATION — {cfg.asset} — {len(result.windows)} folds")
    print(f"{'=' * 70}\n")
    print(
        "Each fold's HMM and regime labels are fit ONLY on that fold's train\n"
        "window, then scored on the following unseen test window. This is a\n"
        "more honest estimate of real-world performance than a single\n"
        "in-sample backtest (see --run-backtest), which fits and evaluates\n"
        "on the same data and will typically look better than it should.\n"
    )

    print("Per-fold results:")
    print(result.per_fold_metrics.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("AGGREGATED OUT-OF-SAMPLE METRICS")
    print(f"{'=' * 70}")
    for k, v in result.aggregated_metrics.items():
        print(f"  {k:30s}: {v:,.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    print(f"\n{DISCLAIMER}\n")
    return result


def cmd_compare_strategies(cfg: Config, force_refresh: bool):
    """Run this project's HMM strategy alongside standard, widely-known
    strategies (buy-and-hold, MA crossover, RSI mean-reversion, MACD
    crossover) on the same data, for an honest side-by-side comparison.

    This is an in-sample comparison (same caveats as --run-backtest); for
    an out-of-sample version, run --walk-forward and compare its
    aggregated_metrics against a similarly walk-forward-validated run of
    each standard strategy (not yet automated here -- see README).
    """
    pipeline = run_pipeline(cfg, force_refresh)
    hmm_result = run_backtest(pipeline["df"], pipeline["regime_names"], pipeline["conf_counts"], cfg)

    standard_comparison = compare_strategies(pipeline["df"], cfg)

    hmm_row = pd.DataFrame([{
        "total_return_pct": hmm_result.metrics["total_return_pct"],
        "benchmark_return_pct": hmm_result.metrics["benchmark_return_pct"],
        "alpha_pct": hmm_result.metrics["alpha_pct"],
        "sharpe": hmm_result.metrics["sharpe"],
        "sortino": hmm_result.metrics["sortino"],
        "max_drawdown_pct": hmm_result.metrics["max_drawdown_pct"],
        "n_trades": hmm_result.metrics["n_trades"],
        "win_rate_pct": hmm_result.metrics["win_rate_pct"],
    }], index=["hmm_regime_strategy (this project)"])

    full_comparison = pd.concat([hmm_row, standard_comparison])
    full_comparison = full_comparison.sort_values("total_return_pct", ascending=False)

    print(f"\n{'=' * 90}")
    print(f"STRATEGY COMPARISON — {cfg.asset} (in-sample)")
    print(f"{'=' * 90}\n")
    print(full_comparison.to_string())
    print(
        "\nThis is an IN-SAMPLE comparison — all strategies are evaluated on the\n"
        "same data they could have been tuned against. A strategy ranking well\n"
        "here is not strong evidence it will perform well going forward. Use\n"
        "--walk-forward for this project's HMM strategy to get an honest\n"
        "out-of-sample estimate, and apply the same discipline before trusting\n"
        "any of the standard strategies' numbers above either.\n"
    )
    print(DISCLAIMER)
    return full_comparison


def cmd_strategy_selector(cfg: Config, force_refresh: bool, n_folds: int, train_frac: float):
    """Walk-forward validate the regime-conditional strategy selector: for
    each fold, learn (from train data only) which of 5 candidate
    strategies historically performed best per regime, then apply that
    choice blind to the following test window. See strategy_selector.py.
    """
    set_seed(cfg.random_state)
    df = load_ohlcv(cfg, force_refresh=force_refresh)

    result = run_selector_walkforward(df, cfg, n_folds=n_folds, train_frac=train_frac)

    print(f"\n{'=' * 70}")
    print(f"STRATEGY SELECTOR — WALK-FORWARD VALIDATION — {cfg.asset} — {len(result.windows)} folds")
    print(f"{'=' * 70}\n")
    print(
        "Each fold learns, from train data only, which candidate strategy\n"
        "(this project's HMM strategy, buy-and-hold, MA crossover, RSI, or\n"
        "MACD) historically performed best in each regime -- then applies\n"
        "that choice to the following unseen test window. This is a\n"
        "meta-strategy, not a self-correcting model: the lookup table is\n"
        "fixed once built per fold and never updated from its own live\n"
        "predictions.\n"
    )
    print("Per-fold results:")
    print(result.per_fold_metrics.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("AGGREGATED OUT-OF-SAMPLE METRICS")
    print(f"{'=' * 70}")
    for k, v in result.aggregated_metrics.items():
        print(f"  {k:30s}: {v:,.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    print(f"\n{DISCLAIMER}\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Regime Terminal CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-backtest", action="store_true")
    group.add_argument("--live-signal", action="store_true")
    group.add_argument("--walk-forward", action="store_true", help="Run honest out-of-sample validation instead of a single in-sample backtest")
    group.add_argument("--compare-strategies", action="store_true", help="Compare this project's HMM strategy against standard strategies (buy-and-hold, MA crossover, RSI, MACD)")
    group.add_argument("--strategy-selector", action="store_true", help="Walk-forward validate a regime-conditional strategy selector (learns best strategy per regime from train data only)")

    parser.add_argument("--asset", default="BTC-USD")
    parser.add_argument("--asset-class", choices=["crypto", "stock"], default="crypto", help="Affects rolling window sizing and annualization (crypto=24/7, stock=~6.5h/day)")
    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--n-components", type=int, default=7)
    parser.add_argument("--hmm-backend", choices=["hmmlearn", "gmm"], default="hmmlearn")
    parser.add_argument("--confirmations", type=int, default=7)
    parser.add_argument("--leverage", type=float, default=2.5)
    parser.add_argument("--min-hold-hours", type=int, default=24)
    parser.add_argument("--cooldown-hours", type=int, default=48)
    parser.add_argument("--trailing-stop", action="store_true")
    parser.add_argument("--aggressive", action="store_true")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass data cache")
    parser.add_argument("--start", default=None, help="(reserved for future use)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of walk-forward folds (only used with --walk-forward)")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Fraction of each fold used for training, rest is test (only used with --walk-forward)")

    args = parser.parse_args()
    cfg = build_config_from_args(args)

    if args.run_backtest:
        cmd_run_backtest(cfg, args.force_refresh)
    elif args.live_signal:
        cmd_live_signal(cfg, args.force_refresh)
    elif args.walk_forward:
        cmd_walkforward(cfg, args.force_refresh, args.n_folds, args.train_frac)
    elif args.compare_strategies:
        cmd_compare_strategies(cfg, args.force_refresh)
    elif args.strategy_selector:
        cmd_strategy_selector(cfg, args.force_refresh, args.n_folds, args.train_frac)


if __name__ == "__main__":
    main()
