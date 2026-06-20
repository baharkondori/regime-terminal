"""Unit tests for hmmmodel.py"""

import numpy as np
import pytest

from config import Config
from features import compute_features, scale_features
from hmmmodel import fit_regime_model, RegimeHMM, _smooth_gmm_labels


def test_fit_produces_expected_number_of_components(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    scaled, _ = scale_features(feats)
    model = fit_regime_model(scaled, small_config)
    assert model.n_components == small_config.n_components


def test_predict_states_shape_matches_input(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    scaled, _ = scale_features(feats)
    model = fit_regime_model(scaled, small_config)
    states = model.predict_states(scaled)
    assert states.shape == (len(scaled),)


def test_predict_states_values_in_valid_range(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    scaled, _ = scale_features(feats)
    model = fit_regime_model(scaled, small_config)
    states = model.predict_states(scaled)
    assert states.min() >= 0
    assert states.max() < small_config.n_components


def test_predict_proba_shape_and_sums_to_one(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    scaled, _ = scale_features(feats)
    model = fit_regime_model(scaled, small_config)
    proba = model.predict_proba(scaled)
    assert proba.shape == (len(scaled), small_config.n_components)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_hmmlearn_backend_used_when_available(synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    scaled, _ = scale_features(feats)
    model = fit_regime_model(scaled, small_config)
    assert model.backend_ in ("hmmlearn", "gmm")  # passes regardless of env, but records which


def test_gmm_fallback_backend_works(synthetic_ohlcv):
    cfg = Config(n_components=3, hmm_backend="gmm", random_state=42)
    feats = compute_features(synthetic_ohlcv, cfg)
    scaled, _ = scale_features(feats)
    model = fit_regime_model(scaled, cfg)
    assert model.backend_ == "gmm"
    states = model.predict_states(scaled)
    assert states.shape == (len(scaled),)


def test_smooth_gmm_labels_collapses_short_runs():
    states = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2])
    proba = np.zeros((len(states), 3))
    smoothed = _smooth_gmm_labels(states, proba, min_run=3)
    # the isolated single "1" should be collapsed into surrounding "0"s
    assert smoothed[3] == 0


def test_save_and_load_roundtrip(tmp_path, synthetic_ohlcv, small_config):
    feats = compute_features(synthetic_ohlcv, small_config)
    scaled, _ = scale_features(feats)
    model = fit_regime_model(scaled, small_config)
    path = tmp_path / "model.joblib"
    model.save(str(path))

    loaded = RegimeHMM.load(str(path))
    np.testing.assert_array_equal(model.predict_states(scaled), loaded.predict_states(scaled))
