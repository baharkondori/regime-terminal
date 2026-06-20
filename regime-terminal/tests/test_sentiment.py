"""Unit tests for sentiment.py

These tests only exercise the pure data-transformation functions
(compute_sentiment_features, merge_sentiment_with_price_features,
asset_to_topic). fetch_sentiment_history() makes a real network call and
is intentionally not unit-tested here -- it requires a live API key and
should be tested manually/in integration once LunarCrush is connected.
"""

import numpy as np
import pandas as pd
import pytest

from sentiment import (
    compute_sentiment_features,
    merge_sentiment_with_price_features,
    asset_to_topic,
    ASSET_TO_TOPIC,
)


def test_asset_to_topic_known_mappings():
    assert asset_to_topic("BTC-USD") == "bitcoin"
    assert asset_to_topic("ETH-USD") == "ethereum"
    assert asset_to_topic("SOL-USD") == "solana"


def test_asset_to_topic_fallback_for_unknown_asset():
    # falls back to stripping "-USD" and lowercasing
    assert asset_to_topic("XRP-USD") == "xrp"


def test_asset_to_topic_mapping_table_is_consistent():
    for asset, topic in ASSET_TO_TOPIC.items():
        assert asset_to_topic(asset) == topic


def test_compute_sentiment_features_requires_sentiment_column():
    bad_df = pd.DataFrame({"galaxy_score": [50, 60]})
    with pytest.raises(ValueError, match="sentiment"):
        compute_sentiment_features(bad_df)


def test_compute_sentiment_features_normalizes_to_expected_range():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    raw = pd.DataFrame({"sentiment": [0, 25, 50, 75, 100] * 2}, index=idx)
    feats = compute_sentiment_features(raw)
    # sentiment=0 -> -1, sentiment=50 -> 0, sentiment=100 -> 1
    assert feats["sentiment_norm"].min() >= -1.001
    assert feats["sentiment_norm"].max() <= 1.001


def test_compute_sentiment_features_50_maps_to_zero():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    raw = pd.DataFrame({"sentiment": [50, 50, 50, 50, 50]}, index=idx)
    feats = compute_sentiment_features(raw)
    np.testing.assert_allclose(feats["sentiment_norm"].values, 0.0, atol=1e-9)


def test_compute_sentiment_features_includes_galaxy_score_when_present():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    raw = pd.DataFrame({"sentiment": [50] * 5, "galaxy_score": [80] * 5}, index=idx)
    feats = compute_sentiment_features(raw)
    assert "galaxy_score_norm" in feats.columns
    np.testing.assert_allclose(feats["galaxy_score_norm"].values, 0.8, atol=1e-9)


def test_compute_sentiment_features_omits_galaxy_score_when_absent():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    raw = pd.DataFrame({"sentiment": [50] * 5}, index=idx)
    feats = compute_sentiment_features(raw)
    assert "galaxy_score_norm" not in feats.columns


def test_compute_sentiment_features_drops_first_row_nan_from_diff():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    raw = pd.DataFrame({"sentiment": [50, 60, 55, 70, 65]}, index=idx)
    feats = compute_sentiment_features(raw)
    # sentiment_change is a diff(), so the very first row has no prior value
    # and should be dropped, not filled with a fabricated 0
    assert len(feats) == len(raw) - 1
    assert not feats.isna().any().any()


def test_merge_sentiment_with_price_features_inner_joins_on_timestamp():
    price_idx = pd.date_range("2024-01-01", periods=10, freq="h")
    sentiment_idx = pd.date_range("2024-01-01 03:00", periods=10, freq="h")  # partial overlap

    price_feats = pd.DataFrame({"returns": np.random.randn(10)}, index=price_idx)
    sentiment_feats = pd.DataFrame({"sentiment_norm": np.random.randn(10)}, index=sentiment_idx)

    combined = merge_sentiment_with_price_features(price_feats, sentiment_feats)
    expected_overlap = price_idx.intersection(sentiment_idx)
    assert len(combined) == len(expected_overlap)
    assert "returns" in combined.columns
    assert "sentiment_norm" in combined.columns


def test_merge_sentiment_with_price_features_no_overlap_returns_empty():
    price_idx = pd.date_range("2024-01-01", periods=5, freq="h")
    sentiment_idx = pd.date_range("2025-01-01", periods=5, freq="h")  # no overlap at all

    price_feats = pd.DataFrame({"returns": np.random.randn(5)}, index=price_idx)
    sentiment_feats = pd.DataFrame({"sentiment_norm": np.random.randn(5)}, index=sentiment_idx)

    combined = merge_sentiment_with_price_features(price_feats, sentiment_feats)
    assert combined.empty


def test_merge_sentiment_with_price_features_no_nans_in_result():
    price_idx = pd.date_range("2024-01-01", periods=20, freq="h")
    sentiment_idx = pd.date_range("2024-01-01 05:00", periods=20, freq="h")

    price_feats = pd.DataFrame({"returns": np.random.randn(20)}, index=price_idx)
    sentiment_feats = pd.DataFrame({"sentiment_norm": np.random.randn(20)}, index=sentiment_idx)

    combined = merge_sentiment_with_price_features(price_feats, sentiment_feats)
    assert not combined.isna().any().any()


def test_fetch_sentiment_history_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("LUNARCRUSH_API_KEY", raising=False)
    from sentiment import fetch_sentiment_history
    with pytest.raises(ValueError, match="API key"):
        fetch_sentiment_history("bitcoin", api_key=None)
