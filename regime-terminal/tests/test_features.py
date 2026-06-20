"""Unit tests for features.py"""

import numpy as np
import pandas as pd

from features import compute_features, scale_features, FEATURE_COLUMNS


def test_compute_features_returns_expected_columns(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    assert list(feats.columns) == FEATURE_COLUMNS


def test_compute_features_no_nans(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    assert not feats.isna().any().any()


def test_compute_features_returns_match_price_changes(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    # log return should be close to pct_change for small moves
    close = synthetic_ohlcv["Close"]
    expected = np.log(close / close.shift(1)).dropna()
    aligned = feats["returns"].reindex(expected.index).dropna()
    np.testing.assert_allclose(aligned.values, expected.loc[aligned.index].values, rtol=1e-8)


def test_compute_features_no_infinite_values(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    assert np.isfinite(feats.values).all()


def test_scale_features_zero_mean_unit_var(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    scaled, scaler = scale_features(feats)
    means = scaled.mean().values
    stds = scaled.std(ddof=0).values
    np.testing.assert_allclose(means, 0.0, atol=1e-6)
    np.testing.assert_allclose(stds, 1.0, atol=1e-2)


def test_scale_features_reuses_fitted_scaler(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    train, test = feats.iloc[:500], feats.iloc[500:]
    _, scaler = scale_features(train)
    test_scaled, scaler_returned = scale_features(test, scaler=scaler)
    assert scaler_returned is scaler
    # Reusing the scaler should NOT re-center test data to its own mean of 0
    assert not np.allclose(test_scaled.mean().values, 0.0, atol=1e-2)
