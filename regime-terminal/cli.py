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
from utils import set_seed


def build_config_from_args(args) -> Config:
    cfg = Config(
        asset=args.asset,
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


def main():
    parser = argparse.ArgumentParser(description="Regime Terminal CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-backtest", action="store_true")
    group.add_argument("--live-signal", action="store_true")

    parser.add_argument("--asset", default="BTC-USD")
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

    args = parser.parse_args()
    cfg = build_config_from_args(args)

    if args.run_backtest:
        cmd_run_backtest(cfg, args.force_refresh)
    elif args.live_signal:
        cmd_live_signal(cfg, args.force_refresh)


if __name__ == "__main__":
    main()
