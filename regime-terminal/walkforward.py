"""
walkforward.py
---------------
Walk-forward validation: the honest alternative to a single in-sample
backtest. Splits historical data into sequential (train, test) windows --
fit the HMM and labels on the train window only, then run the backtest on
the *following*, unseen test window. Repeat across the whole dataset and
aggregate results.

Why this matters (this is the rigor the README has been recommending all
along, finally implemented as code rather than just advice):

A single backtest that fits the HMM and evaluates the strategy on the same
historical window will almost always look better than the strategy
actually performs going forward, because the regime labels and thresholds
were implicitly "informed" by the very data being scored. Walk-forward
testing fits each window blind to what comes next, which is a much more
honest estimate of how the strategy would have performed in real time.

This does NOT eliminate the deeper risk that markets change over time in
ways no amount of backtesting can anticipate -- it only removes the
specific, avoidable bias of testing on data the model implicitly already
"saw" during fitting.

Public API:
    class WalkForwardWindow (one fold's train range, test range, and result)
    class WalkForwardResult (all folds + aggregated summary)
    run_walkforward(df, cfg, n_folds=5, train_frac=0.7) -> WalkForwardResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from config import Config
from features import compute_features, scale_features
from hmmmodel import fit_regime_model
from regimelabeler import label_regimes, apply_labels
from strategies import compute_indicators, confirmations_count_series
from backtester import run_backtest, BacktestResult


@dataclass
class WalkForwardWindow:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: BacktestResult


@dataclass
class WalkForwardResult:
    windows: List[WalkForwardWindow] = field(default_factory=list)
    aggregated_metrics: dict = field(default_factory=dict)
    per_fold_metrics: pd.DataFrame = None


def _build_pipeline_for_window(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config):
    """Fit the HMM + labels on train_df only, then compute regimes and
    confirmations for test_df using that already-fitted model. test_df is
    never used to fit anything -- it's scored blind."""
    train_feats = compute_features(train_df, cfg)
    train_scaled, scaler = scale_features(train_feats)

    model = fit_regime_model(train_scaled, cfg)
    train_states = model.predict_states(train_scaled)
    _, mapping = label_regimes(train_feats["returns"], train_states, cfg)

    # Apply the train-fitted scaler and model to the test window -- this is
    # the critical part that makes this "walk-forward" rather than just
    # re-fitting on everything: the test window's features are scaled with
    # train statistics, and classified with the train-fitted model, exactly
    # as if this were genuinely unseen future data at decision time.
    test_feats = compute_features(test_df, cfg)
    if test_feats.empty:
        return None

    test_scaled, _ = scale_features(test_feats, scaler=scaler)
    test_states = model.predict_states(test_scaled)
    test_regime_names = apply_labels(test_states, mapping)

    test_aligned_df = test_df.loc[test_feats.index]
    test_indicators = compute_indicators(test_aligned_df, cfg)
    common_idx = test_indicators.dropna().index.intersection(test_feats.index)

    if len(common_idx) < 10:
        return None  # not enough usable bars in this test window

    test_aligned_df = test_aligned_df.loc[common_idx]
    test_regime_names_aligned = pd.Series(test_regime_names, index=test_feats.index).loc[common_idx].values
    test_indicators_aligned = test_indicators.loc[common_idx]
    test_conf_counts = confirmations_count_series(test_indicators_aligned, "bull", cfg)

    return test_aligned_df, test_regime_names_aligned, test_conf_counts


def run_walkforward(
    df: pd.DataFrame,
    cfg: Config,
    n_folds: int = 5,
    train_frac: float = 0.7,
) -> WalkForwardResult:
    """Run walk-forward validation across n_folds sequential, non-overlapping
    test windows, each preceded by its own train window.

    Parameters
    ----------
    df : full OHLCV DataFrame to validate against
    cfg : Config (n_components, leverage, confirmations_required, etc.)
    n_folds : how many sequential test windows to create
    train_frac : fraction of each fold's span used for training (the rest
                 is the test window that follows it)

    Returns
    -------
    WalkForwardResult with per-fold results and aggregated out-of-sample
    metrics (computed by concatenating all test-window equity curves into
    one continuous series, so compounding works correctly across folds).
    """
    n = len(df)
    fold_size = n // n_folds
    if fold_size < 50:
        raise ValueError(
            f"Not enough data for {n_folds} folds (only {n} bars total, "
            f"{fold_size} per fold). Use fewer folds or more data."
        )

    windows: List[WalkForwardWindow] = []
    all_test_equity_segments = []
    all_test_benchmark_segments = []
    running_equity = cfg.initial_equity
    running_benchmark = cfg.initial_equity

    for fold_i in range(n_folds):
        fold_start = fold_i * fold_size
        fold_end = min((fold_i + 1) * fold_size, n)
        train_end_pos = fold_start + int((fold_end - fold_start) * train_frac)

        train_df = df.iloc[fold_start:train_end_pos]
        test_df = df.iloc[train_end_pos:fold_end]

        if len(train_df) < 50 or len(test_df) < 10:
            continue  # skip degenerate folds rather than error on them

        built = _build_pipeline_for_window(train_df, test_df, cfg)
        if built is None:
            continue
        test_aligned_df, test_regime_names, test_conf_counts = built

        # Each fold's backtest starts fresh at the running equity carried
        # over from the previous fold, so the aggregated curve compounds
        # correctly across the whole walk-forward run rather than each
        # fold resetting to cfg.initial_equity independently.
        fold_cfg = cfg
        fold_initial_equity = running_equity
        original_initial_equity = cfg.initial_equity
        import copy
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.initial_equity = fold_initial_equity

        result = run_backtest(test_aligned_df, test_regime_names, test_conf_counts, fold_cfg)

        running_equity = result.equity_curve.iloc[-1] if len(result.equity_curve) else running_equity
        # Benchmark also compounds across folds for a fair total comparison
        fold_benchmark_return = (
            test_aligned_df["Close"].iloc[-1] / test_aligned_df["Close"].iloc[0]
            if len(test_aligned_df) else 1.0
        )
        running_benchmark = running_benchmark * fold_benchmark_return

        all_test_equity_segments.append(result.equity_curve)
        all_test_benchmark_segments.append(result.benchmark_curve * (running_benchmark / result.benchmark_curve.iloc[-1]) if len(result.benchmark_curve) else result.benchmark_curve)

        windows.append(
            WalkForwardWindow(
                fold_index=fold_i,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0] if len(test_df) else None,
                test_end=test_df.index[-1] if len(test_df) else None,
                result=result,
            )
        )

    if not windows:
        raise ValueError("No valid folds produced -- try fewer folds or check your data.")

    aggregated = _aggregate_walkforward(windows, cfg)
    per_fold_df = _per_fold_summary(windows)

    return WalkForwardResult(
        windows=windows,
        aggregated_metrics=aggregated,
        per_fold_metrics=per_fold_df,
    )


def _per_fold_summary(windows: List[WalkForwardWindow]) -> pd.DataFrame:
    rows = []
    for w in windows:
        m = w.result.metrics
        rows.append({
            "fold": w.fold_index,
            "train_start": w.train_start,
            "train_end": w.train_end,
            "test_start": w.test_start,
            "test_end": w.test_end,
            "test_return_pct": m["total_return_pct"],
            "benchmark_return_pct": m["benchmark_return_pct"],
            "alpha_pct": m["alpha_pct"],
            "sharpe": m["sharpe"],
            "n_trades": m["n_trades"],
            "win_rate_pct": m["win_rate_pct"],
            "max_drawdown_pct": m["max_drawdown_pct"],
        })
    return pd.DataFrame(rows)


def _aggregate_walkforward(windows: List[WalkForwardWindow], cfg: Config) -> dict:
    """Aggregate out-of-sample performance across all folds. Total return is
    computed by chaining each fold's return (since equity compounds fold to
    fold), while trade counts/win rate are summed/averaged across folds."""
    total_multiplier = 1.0
    benchmark_multiplier = 1.0
    total_trades = 0
    total_wins = 0
    all_drawdowns = []
    all_sharpes = []

    for w in windows:
        m = w.result.metrics
        total_multiplier *= (1 + m["total_return_pct"] / 100)
        benchmark_multiplier *= (1 + m["benchmark_return_pct"] / 100)
        total_trades += m["n_trades"]
        total_wins += round(m["n_trades"] * m["win_rate_pct"] / 100)
        all_drawdowns.append(m["max_drawdown_pct"])
        all_sharpes.append(m["sharpe"])

    out_of_sample_return_pct = (total_multiplier - 1) * 100
    out_of_sample_benchmark_pct = (benchmark_multiplier - 1) * 100

    return {
        "n_folds": len(windows),
        "out_of_sample_return_pct": out_of_sample_return_pct,
        "out_of_sample_benchmark_pct": out_of_sample_benchmark_pct,
        "out_of_sample_alpha_pct": out_of_sample_return_pct - out_of_sample_benchmark_pct,
        "total_trades": total_trades,
        "overall_win_rate_pct": 100 * total_wins / total_trades if total_trades > 0 else 0.0,
        "worst_fold_drawdown_pct": min(all_drawdowns) if all_drawdowns else 0.0,
        "avg_fold_sharpe": float(np.mean(all_sharpes)) if all_sharpes else 0.0,
    }


if __name__ == "__main__":
    import pandas as pd

    cfg = Config(n_components=4)
    df = pd.read_parquet(".cache/synthetic_test_data.parquet")

    result = run_walkforward(df, cfg, n_folds=5, train_frac=0.7)

    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD VALIDATION — {len(result.windows)} folds")
    print(f"{'=' * 70}\n")
    print("Per-fold results:")
    print(result.per_fold_metrics.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("AGGREGATED OUT-OF-SAMPLE METRICS")
    print(f"{'=' * 70}")
    for k, v in result.aggregated_metrics.items():
        print(f"  {k:30s}: {v:,.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    print(
        "\nNote: these are out-of-sample results -- each fold's HMM and "
        "regime labels were fit only on that fold's train window, then "
        "scored on the following unseen test window. This is a more "
        "honest estimate of real-world performance than a single "
        "in-sample backtest, though it still cannot account for the "
        "market regime changing in ways never seen in this dataset at all."
    )
