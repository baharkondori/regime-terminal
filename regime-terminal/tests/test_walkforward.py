"""Unit tests for walkforward.py"""

import numpy as np
import pandas as pd
import pytest

from config import Config
from walkforward import run_walkforward, WalkForwardResult


@pytest.fixture
def long_synthetic_ohlcv():
    """A longer synthetic series than the standard fixture, since
    walk-forward needs enough bars to support multiple train/test folds."""
    rng = np.random.RandomState(7)
    n = 3000  # several "regimes" worth of hourly bars

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

    df = pd.DataFrame(
        {"Open": openp, "High": high, "Low": low, "Close": price, "Volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def wf_config():
    return Config(n_components=3, hmm_n_iter=100, confirmations_required=5, random_state=42)


def test_run_walkforward_returns_result_with_expected_structure(long_synthetic_ohlcv, wf_config):
    result = run_walkforward(long_synthetic_ohlcv, wf_config, n_folds=3, train_frac=0.7)
    assert isinstance(result, WalkForwardResult)
    assert len(result.windows) > 0
    assert isinstance(result.per_fold_metrics, pd.DataFrame)
    assert isinstance(result.aggregated_metrics, dict)


def test_walkforward_folds_are_sequential_and_non_overlapping(long_synthetic_ohlcv, wf_config):
    result = run_walkforward(long_synthetic_ohlcv, wf_config, n_folds=3, train_frac=0.7)
    for i in range(len(result.windows) - 1):
        current_test_end = result.windows[i].test_end
        next_train_start = result.windows[i + 1].train_start
        # the next fold's train window should start at or after this fold's test window ends
        assert next_train_start >= current_test_end


def test_walkforward_test_window_never_used_for_fitting(long_synthetic_ohlcv, wf_config):
    """The core walk-forward guarantee: each fold's train_end must come
    strictly before its own test_start -- i.e. the model never sees the
    data it's about to be scored on."""
    result = run_walkforward(long_synthetic_ohlcv, wf_config, n_folds=3, train_frac=0.7)
    for w in result.windows:
        assert w.train_end < w.test_start


def test_walkforward_aggregated_metrics_has_expected_keys(long_synthetic_ohlcv, wf_config):
    result = run_walkforward(long_synthetic_ohlcv, wf_config, n_folds=3, train_frac=0.7)
    expected_keys = {
        "n_folds", "out_of_sample_return_pct", "out_of_sample_benchmark_pct",
        "out_of_sample_alpha_pct", "total_trades", "overall_win_rate_pct",
        "worst_fold_drawdown_pct", "avg_fold_sharpe",
    }
    assert expected_keys.issubset(set(result.aggregated_metrics.keys()))


def test_walkforward_per_fold_metrics_row_count_matches_windows(long_synthetic_ohlcv, wf_config):
    result = run_walkforward(long_synthetic_ohlcv, wf_config, n_folds=3, train_frac=0.7)
    assert len(result.per_fold_metrics) == len(result.windows)


def test_walkforward_raises_with_too_many_folds_for_data_size(wf_config):
    tiny_idx = pd.date_range("2024-01-01", periods=20, freq="h")
    tiny_df = pd.DataFrame(
        {"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 1000},
        index=tiny_idx,
    )
    with pytest.raises(ValueError):
        run_walkforward(tiny_df, wf_config, n_folds=10, train_frac=0.7)


def test_walkforward_total_trades_is_sum_across_folds(long_synthetic_ohlcv, wf_config):
    result = run_walkforward(long_synthetic_ohlcv, wf_config, n_folds=3, train_frac=0.7)
    manual_sum = sum(w.result.metrics["n_trades"] for w in result.windows)
    assert result.aggregated_metrics["total_trades"] == manual_sum


def test_walkforward_different_train_frac_changes_window_sizes(long_synthetic_ohlcv, wf_config):
    result_70 = run_walkforward(long_synthetic_ohlcv, wf_config, n_folds=3, train_frac=0.7)
    result_50 = run_walkforward(long_synthetic_ohlcv, wf_config, n_folds=3, train_frac=0.5)
    # With a smaller train_frac, the train window should be shorter (and test window longer)
    train_span_70 = result_70.windows[0].train_end - result_70.windows[0].train_start
    train_span_50 = result_50.windows[0].train_end - result_50.windows[0].train_start
    assert train_span_50 < train_span_70
