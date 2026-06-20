"""Unit tests for prediction_log.py"""

import os

import numpy as np
import pandas as pd
import pytest

from prediction_log import log_snapshot, load_log, grade_log, summarize_accuracy, LOG_COLUMNS


@pytest.fixture
def tmp_log_path(tmp_path):
    return str(tmp_path / "signal_log.csv")


@pytest.fixture
def sample_df_and_regimes():
    idx = pd.date_range("2024-01-01", periods=50, freq="h")
    regimes = np.array(["weak_bull"] * 10 + ["bull"] * 10 + ["weak_bear"] * 10 + ["bull"] * 10 + ["chop"] * 10)
    df = pd.DataFrame({"Close": np.arange(100, 150)}, index=idx)
    return df, regimes, idx


def test_log_snapshot_creates_file_with_one_row(tmp_log_path):
    log_snapshot(
        tmp_log_path, asset="BTC-USD", bar_timestamp="2024-01-01 05:00:00",
        regime="weak_bull", confidence=0.94, conf_count=4, conf_required=7,
        action="WATCH", price_at_log=105.0, predicted_next_regime="bull",
        predicted_next_prob=0.79, horizon_bars=10,
    )
    log = load_log(tmp_log_path)
    assert len(log) == 1
    assert log.iloc[0]["regime"] == "weak_bull"
    assert set(LOG_COLUMNS).issubset(set(log.columns))


def test_log_snapshot_skips_exact_duplicate_bar(tmp_log_path):
    kwargs = dict(
        asset="BTC-USD", bar_timestamp="2024-01-01 05:00:00", regime="weak_bull",
        confidence=0.94, conf_count=4, conf_required=7, action="WATCH",
        price_at_log=105.0, predicted_next_regime="bull", predicted_next_prob=0.79,
        horizon_bars=10,
    )
    log_snapshot(tmp_log_path, **kwargs)
    log_snapshot(tmp_log_path, **kwargs)
    log = load_log(tmp_log_path)
    assert len(log) == 1


def test_load_log_returns_empty_dataframe_if_file_missing(tmp_log_path):
    log = load_log(tmp_log_path)
    assert log.empty
    assert set(LOG_COLUMNS).issubset(set(log.columns))


def test_grade_log_marks_correct_prediction(tmp_log_path, sample_df_and_regimes):
    df, regimes, idx = sample_df_and_regimes
    # bar 5 (weak_bull) + horizon 10 = bar 15, which IS in the "bull" segment (10-19)
    log_snapshot(tmp_log_path, "BTC-USD", idx[5], "weak_bull", 0.94, 4, 7,
                 "WATCH", 105.0, "bull", 0.79, horizon_bars=10)
    graded = grade_log(tmp_log_path, df, regimes)
    assert graded.iloc[0]["outcome_known"] == True  # noqa: E712
    assert graded.iloc[0]["prediction_correct"] == True  # noqa: E712
    assert graded.iloc[0]["actual_regime_at_horizon"] == "bull"


def test_grade_log_marks_incorrect_prediction(tmp_log_path, sample_df_and_regimes):
    df, regimes, idx = sample_df_and_regimes
    # bar 5 (weak_bull) + horizon 15 = bar 20, which is in "weak_bear" (20-29), NOT "bull"
    log_snapshot(tmp_log_path, "BTC-USD", idx[5], "weak_bull", 0.94, 4, 7,
                 "WATCH", 105.0, "bull", 0.79, horizon_bars=15)
    graded = grade_log(tmp_log_path, df, regimes)
    assert graded.iloc[0]["outcome_known"] == True  # noqa: E712
    assert graded.iloc[0]["prediction_correct"] == False  # noqa: E712
    assert graded.iloc[0]["actual_regime_at_horizon"] == "weak_bear"


def test_grade_log_leaves_future_predictions_ungraded(tmp_log_path, sample_df_and_regimes):
    df, regimes, idx = sample_df_and_regimes
    # bar 45 + horizon 10 = bar 55, which doesn't exist (only 50 bars total)
    log_snapshot(tmp_log_path, "BTC-USD", idx[45], "chop", 0.5, 2, 7,
                 "WATCH", 145.0, "bull", 0.5, horizon_bars=10)
    graded = grade_log(tmp_log_path, df, regimes)
    assert graded.iloc[0]["outcome_known"] == False  # noqa: E712
    assert pd.isna(graded.iloc[0]["prediction_correct"]) or graded.iloc[0]["prediction_correct"] is None


def test_grade_log_skips_rows_with_no_prediction(tmp_log_path, sample_df_and_regimes):
    df, regimes, idx = sample_df_and_regimes
    log_snapshot(tmp_log_path, "BTC-USD", idx[5], "chop", 0.5, 2, 7,
                 "WATCH", 105.0, None, None, horizon_bars=10)
    graded = grade_log(tmp_log_path, df, regimes)
    assert graded.iloc[0]["outcome_known"] == False  # noqa: E712


def test_grade_log_is_idempotent(tmp_log_path, sample_df_and_regimes):
    """Calling grade_log multiple times should not change already-graded rows."""
    df, regimes, idx = sample_df_and_regimes
    log_snapshot(tmp_log_path, "BTC-USD", idx[5], "weak_bull", 0.94, 4, 7,
                 "WATCH", 105.0, "bull", 0.79, horizon_bars=10)
    graded_once = grade_log(tmp_log_path, df, regimes)
    graded_twice = grade_log(tmp_log_path, df, regimes)
    assert graded_once.iloc[0]["prediction_correct"] == graded_twice.iloc[0]["prediction_correct"]


def test_summarize_accuracy_with_mixed_results(tmp_log_path, sample_df_and_regimes):
    df, regimes, idx = sample_df_and_regimes
    # one correct, one incorrect, one ungraded
    log_snapshot(tmp_log_path, "BTC-USD", idx[5], "weak_bull", 0.94, 4, 7,
                 "WATCH", 105.0, "bull", 0.79, horizon_bars=10)   # correct
    log_snapshot(tmp_log_path, "BTC-USD", idx[6], "weak_bull", 0.94, 4, 7,
                 "WATCH", 106.0, "bull", 0.79, horizon_bars=20)   # likely incorrect (lands in weak_bear or chop)
    log_snapshot(tmp_log_path, "BTC-USD", idx[45], "chop", 0.5, 2, 7,
                 "WATCH", 145.0, "bull", 0.5, horizon_bars=10)    # ungraded
    graded = grade_log(tmp_log_path, df, regimes)
    summary = summarize_accuracy(graded)
    assert summary["n_graded"] == 2
    assert summary["n_pending"] == 1
    assert summary["accuracy_pct"] is not None


def test_summarize_accuracy_handles_empty_log():
    empty_df = pd.DataFrame(columns=LOG_COLUMNS)
    summary = summarize_accuracy(empty_df)
    assert summary["n_graded"] == 0
    assert summary["accuracy_pct"] is None
