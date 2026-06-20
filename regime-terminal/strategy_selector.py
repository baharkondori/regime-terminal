"""
strategy_selector.py
---------------------
A meta-strategy: instead of committing to one fixed strategy, learn which
of several candidate strategies (this project's HMM strategy, plus the
standard strategies in benchmark_strategies.py) historically performed
best *within each regime*, then apply that regime-conditional choice
going forward.

This is fundamentally different from -- and meaningfully safer than --
a self-correcting/reinforcement-learning approach: the "learning" here is
a simple, auditable lookup table (regime -> best historical strategy),
built once per walk-forward fold from train-window data only, never
updated based on its own live predictions. You can print the table and
see exactly why a choice was made; there's no opaque weight-adjustment
process to silently drift in a bad direction.

Honest framing: this is still real overfitting risk, just a different
shape of it. With ~7 regimes and 5 candidate strategies, it's easy for a
regime to have very few historical occurrences in a given training
window, making "the best strategy for this regime" a noisy, small-sample
choice rather than a robust pattern. min_observations_per_regime exists
specifically to guard against trusting a regime->strategy mapping built
from too little data -- regimes below that threshold fall back to a
neutral default (buy-and-hold) rather than a possibly-spurious "winner."

This is validated walk-forward from the start (see run_selector_walkforward):
each fold's regime->strategy lookup table is built ONLY from that fold's
train window, then applied blind to the following test window, exactly
like walkforward.py does for the HMM strategy alone.

Public API:
    build_regime_strategy_table(df, regime_names, cfg, min_observations_per_regime=30) -> pd.DataFrame
    apply_strategy_table(df, regime_names, strategy_table, cfg) -> np.ndarray[bool]  (position signal)
    run_selector_walkforward(df, cfg, n_folds=5, train_frac=0.7, min_observations_per_regime=30) -> WalkForwardResult
"""

from __future__ import annotations

import copy
from typing import List

import numpy as np
import pandas as pd

from config import Config
from benchmark_strategies import (
    moving_average_crossover_signal,
    rsi_mean_reversion_signal,
    macd_crossover_signal,
    buy_and_hold_signal,
    run_simple_backtest,
)
from walkforward import (
    _build_pipeline_for_window,
    WalkForwardWindow,
    WalkForwardResult,
    _per_fold_summary,
    _aggregate_walkforward,
)

DEFAULT_FALLBACK_STRATEGY = "buy_and_hold"

# Candidate strategies this selector chooses between. "hmm_regime_strategy"
# is handled specially (it needs regime_names + conf_counts, not just a
# boolean signal function), the rest are simple signal functions from
# benchmark_strategies.py.
SIGNAL_STRATEGIES = {
    "buy_and_hold": buy_and_hold_signal,
    "ma_crossover_50_200": lambda df: moving_average_crossover_signal(df, fast=50, slow=200),
    "rsi_mean_reversion": lambda df: rsi_mean_reversion_signal(df, period=14, oversold=30, overbought=70),
    "macd_crossover": lambda df: macd_crossover_signal(df, fast=12, slow=26, signal=9),
}
ALL_CANDIDATE_NAMES = ["hmm_regime_strategy"] + list(SIGNAL_STRATEGIES.keys())


def _per_bar_strategy_returns(
    df: pd.DataFrame,
    regime_names: np.ndarray,
    conf_counts: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    """Compute each candidate strategy's bar-to-bar position signal and the
    resulting next-bar return contribution, so we can attribute performance
    to whichever regime was active at that bar.

    Returns a DataFrame indexed like df, with one column per strategy name
    holding that bar's forward return if the strategy was in position
    (0.0 if flat), plus a 'regime' column. This is intentionally simpler
    than running the full Trade-based backtester per strategy: for building
    the regime->strategy lookup table we only need relative performance
    per regime, not trade-level bookkeeping.
    """
    returns = df["Close"].pct_change().shift(-1).fillna(0.0)  # next-bar return from holding at this bar

    out = pd.DataFrame(index=df.index)
    out["regime"] = regime_names

    hmm_in_position = np.array([
        regime_names[i] in ("weak_bull", "bull", "strong_bull") and conf_counts[i] >= cfg.confirmations_required
        for i in range(len(df))
    ])
    out["hmm_regime_strategy"] = np.where(hmm_in_position, returns.values, 0.0)

    for name, signal_fn in SIGNAL_STRATEGIES.items():
        signal = signal_fn(df)
        out[name] = np.where(signal, returns.values, 0.0)

    return out


def build_regime_strategy_table(
    df: pd.DataFrame,
    regime_names: np.ndarray,
    conf_counts: np.ndarray,
    cfg: Config,
    min_observations_per_regime: int = 30,
) -> pd.DataFrame:
    """Build the regime -> best-historical-strategy lookup table from a
    single (typically train-window) dataset.

    For each regime observed, sum each candidate strategy's per-bar return
    contribution (see _per_bar_strategy_returns) across every bar where
    that regime was active, then pick the strategy with the highest total.
    Regimes with fewer than min_observations_per_regime bars fall back to
    DEFAULT_FALLBACK_STRATEGY rather than trusting a small-sample "winner".

    Returns a DataFrame indexed by regime name with columns:
        best_strategy, n_observations, and one column per candidate
        strategy showing its total summed return in that regime (so you
        can see exactly how close/lopsided each choice was, not just the
        winner).
    """
    per_bar = _per_bar_strategy_returns(df, regime_names, conf_counts, cfg)
    strategy_cols = ALL_CANDIDATE_NAMES

    grouped = per_bar.groupby("regime")[strategy_cols].sum()
    counts = per_bar.groupby("regime").size()

    rows = []
    for regime in grouped.index:
        n_obs = int(counts[regime])
        if n_obs < min_observations_per_regime:
            best = DEFAULT_FALLBACK_STRATEGY
        else:
            best = grouped.loc[regime, strategy_cols].idxmax()
        row = {"regime": regime, "best_strategy": best, "n_observations": n_obs}
        for col in strategy_cols:
            row[col] = grouped.loc[regime, col]
        rows.append(row)

    table = pd.DataFrame(rows).set_index("regime")
    return table


def apply_strategy_table(
    df: pd.DataFrame,
    regime_names: np.ndarray,
    conf_counts: np.ndarray,
    strategy_table: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    """Given a regime->best_strategy lookup table (built on train data),
    apply it to (typically test-window) data to produce a single combined
    boolean position signal: at each bar, use whichever strategy the table
    says is best for that bar's regime.

    Regimes present in this data but absent from strategy_table (e.g. a
    regime that never occurred during training) fall back to
    DEFAULT_FALLBACK_STRATEGY.
    """
    n = len(df)
    combined_signal = np.zeros(n, dtype=bool)

    hmm_in_position = np.array([
        regime_names[i] in ("weak_bull", "bull", "strong_bull") and conf_counts[i] >= cfg.confirmations_required
        for i in range(n)
    ])
    signal_cache = {"hmm_regime_strategy": hmm_in_position}
    for name, signal_fn in SIGNAL_STRATEGIES.items():
        signal_cache[name] = signal_fn(df)

    for i in range(n):
        regime = regime_names[i]
        if regime in strategy_table.index:
            chosen = strategy_table.loc[regime, "best_strategy"]
        else:
            chosen = DEFAULT_FALLBACK_STRATEGY
        combined_signal[i] = signal_cache[chosen][i]

    return combined_signal


def run_selector_walkforward(
    df: pd.DataFrame,
    cfg: Config,
    n_folds: int = 5,
    train_frac: float = 0.7,
    min_observations_per_regime: int = 30,
) -> WalkForwardResult:
    """Walk-forward validation of the regime->strategy selector itself.

    For each fold: fit the HMM on the train window (via the same
    _build_pipeline_for_window helper walkforward.py uses), build the
    regime->strategy lookup table from train-window performance only, then
    apply that table to the test window and score it with
    run_simple_backtest(). The lookup table never sees test-window data
    before being applied to it.
    """
    n = len(df)
    fold_size = n // n_folds
    if fold_size < 50:
        raise ValueError(
            f"Not enough data for {n_folds} folds (only {n} bars total, "
            f"{fold_size} per fold). Use fewer folds or more data."
        )

    windows: List[WalkForwardWindow] = []
    running_equity = cfg.initial_equity

    for fold_i in range(n_folds):
        fold_start = fold_i * fold_size
        fold_end = min((fold_i + 1) * fold_size, n)
        train_end_pos = fold_start + int((fold_end - fold_start) * train_frac)

        train_df = df.iloc[fold_start:train_end_pos]
        test_df = df.iloc[train_end_pos:fold_end]

        if len(train_df) < 50 or len(test_df) < 10:
            continue

        from features import compute_features, scale_features
        from hmmmodel import fit_regime_model
        from regimelabeler import label_regimes, apply_labels
        from strategies import compute_indicators, confirmations_count_series

        train_feats = compute_features(train_df, cfg)
        train_scaled, scaler = scale_features(train_feats)
        model = fit_regime_model(train_scaled, cfg)
        train_states = model.predict_states(train_scaled)
        _, mapping = label_regimes(train_feats["returns"], train_states, cfg)
        train_regime_names = apply_labels(train_states, mapping)

        train_aligned_df = train_df.loc[train_feats.index]
        train_indicators = compute_indicators(train_aligned_df, cfg)
        train_common_idx = train_indicators.dropna().index.intersection(train_feats.index)
        train_aligned_df = train_aligned_df.loc[train_common_idx]
        train_regime_names_aligned = pd.Series(train_regime_names, index=train_feats.index).loc[train_common_idx].values
        train_conf_counts = confirmations_count_series(train_indicators.loc[train_common_idx], "bull", cfg)

        strategy_table = build_regime_strategy_table(
            train_aligned_df, train_regime_names_aligned, train_conf_counts, cfg,
            min_observations_per_regime=min_observations_per_regime,
        )

        built = _build_pipeline_for_window(train_df, test_df, cfg)
        if built is None:
            continue
        test_aligned_df, test_regime_names, test_conf_counts = built

        test_signal = apply_strategy_table(
            test_aligned_df, test_regime_names, test_conf_counts, strategy_table, cfg
        )

        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.initial_equity = running_equity
        result = run_simple_backtest(test_aligned_df, test_signal, fold_cfg)
        running_equity = result.equity_curve.iloc[-1] if len(result.equity_curve) else running_equity

        windows.append(WalkForwardWindow(
            fold_index=fold_i,
            train_start=train_df.index[0], train_end=train_df.index[-1],
            test_start=test_df.index[0] if len(test_df) else None,
            test_end=test_df.index[-1] if len(test_df) else None,
            result=result,
        ))

    if not windows:
        raise ValueError("No valid folds produced -- try fewer folds or check your data.")

    aggregated = _aggregate_walkforward(windows, cfg)
    per_fold_df = _per_fold_summary(windows)

    return WalkForwardResult(windows=windows, aggregated_metrics=aggregated, per_fold_metrics=per_fold_df)


if __name__ == "__main__":
    import pandas as pd

    cfg = Config(n_components=4)
    df = pd.read_parquet(".cache/synthetic_test_data.parquet")

    print("Building a single regime->strategy table on the full dataset (for illustration only --")
    print("the real validation below never lets the table see test data before scoring it):\n")

    from features import compute_features, scale_features
    from hmmmodel import fit_regime_model
    from regimelabeler import label_regimes, apply_labels
    from strategies import compute_indicators, confirmations_count_series

    feats = compute_features(df, cfg)
    scaled, scaler = scale_features(feats)
    model = fit_regime_model(scaled, cfg)
    states = model.predict_states(scaled)
    _, mapping = label_regimes(feats["returns"], states, cfg)
    regime_names = apply_labels(states, mapping)
    aligned_df = df.loc[feats.index]
    indicators = compute_indicators(aligned_df, cfg)
    common_idx = indicators.dropna().index.intersection(feats.index)
    aligned_df = aligned_df.loc[common_idx]
    regime_names_aligned = pd.Series(regime_names, index=feats.index).loc[common_idx].values
    conf_counts = confirmations_count_series(indicators.loc[common_idx], "bull", cfg)

    table = build_regime_strategy_table(aligned_df, regime_names_aligned, conf_counts, cfg)
    print(table.to_string())

    print(f"\n{'=' * 70}")
    print("WALK-FORWARD VALIDATION OF THE STRATEGY SELECTOR")
    print(f"{'=' * 70}\n")
    result = run_selector_walkforward(df, cfg, n_folds=5, train_frac=0.7)
    print(result.per_fold_metrics.to_string(index=False))
    print("\nAggregated:")
    for k, v in result.aggregated_metrics.items():
        print(f"  {k:30s}: {v:,.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")
