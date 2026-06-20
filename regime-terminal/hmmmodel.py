"""
hmmmodel.py
-----------
Trains and evaluates the regime-detection model. Prefers hmmlearn's
GaussianHMM (a true Hidden Markov Model with a learned state-transition
matrix), and falls back to sklearn.mixture.GaussianMixture with a simple
Viterbi-like smoothing pass if hmmlearn is unavailable in the environment.

Public API:
    class RegimeHMM
        .fit(features_df)
        .predict_states(features_df) -> np.ndarray[int]
        .predict_proba(features_df) -> np.ndarray[T, n_components]
        .save(path) / RegimeHMM.load(path)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib

from config import Config

try:
    from hmmlearn.hmm import GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False

from sklearn.mixture import GaussianMixture


def _smooth_gmm_labels(states: np.ndarray, proba: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Cheap Viterbi-like smoothing for the GMM fallback: collapse runs
    shorter than `min_run` into the surrounding state, since plain GMM
    has no transition matrix and can flicker between states bar-to-bar."""
    states = states.copy()
    n = len(states)
    i = 0
    while i < n:
        j = i
        while j < n and states[j] == states[i]:
            j += 1
        run_len = j - i
        if run_len < min_run and i > 0:
            states[i:j] = states[i - 1]
        i = j
    return states


@dataclass
class RegimeHMM:
    cfg: Config
    backend_: str = ""
    model_: object = None
    n_components: int = 7

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_components = cfg.n_components
        self.model_ = None
        self.backend_ = ""

    def fit(self, features_df: pd.DataFrame) -> "RegimeHMM":
        X = features_df.values
        use_hmmlearn = self.cfg.hmm_backend == "hmmlearn" and _HMMLEARN_AVAILABLE

        if self.cfg.hmm_backend == "hmmlearn" and not _HMMLEARN_AVAILABLE:
            warnings.warn(
                "hmmlearn not available in this environment; falling back to "
                "sklearn.mixture.GaussianMixture. Install hmmlearn for true "
                "HMM transition-matrix modeling: pip install hmmlearn"
            )

        if use_hmmlearn:
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.cfg.hmm_covariance_type,
                n_iter=self.cfg.hmm_n_iter,
                tol=self.cfg.hmm_tol,
                verbose=self.cfg.hmm_verbose,
                random_state=self.cfg.random_state,
            )
            model.fit(X)
            self.model_ = model
            self.backend_ = "hmmlearn"
        else:
            model = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.cfg.hmm_covariance_type
                if self.cfg.hmm_covariance_type in ("full", "tied", "diag", "spherical")
                else "full",
                n_init=3,
                random_state=self.cfg.random_state,
                max_iter=self.cfg.hmm_n_iter,
            )
            model.fit(X)
            self.model_ = model
            self.backend_ = "gmm"

        return self

    def predict_states(self, features_df: pd.DataFrame) -> np.ndarray:
        X = features_df.values
        if self.backend_ == "hmmlearn":
            return self.model_.predict(X)
        states = self.model_.predict(X)
        return _smooth_gmm_labels(states, self.predict_proba(features_df))

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        X = features_df.values
        if self.backend_ == "hmmlearn":
            # hmmlearn's predict_proba gives posterior state probabilities
            # via the forward-backward algorithm.
            return self.model_.predict_proba(X)
        return self.model_.predict_proba(X)

    def score_samples(self, features_df: pd.DataFrame):
        """Log-likelihood of the data under the fitted model (model fit quality)."""
        X = features_df.values
        if self.backend_ == "hmmlearn":
            return self.model_.score(X)
        return self.model_.score(X) * len(X)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "RegimeHMM":
        return joblib.load(path)


def fit_regime_model(features_df: pd.DataFrame, cfg: Config) -> RegimeHMM:
    """Convenience wrapper: instantiate + fit a RegimeHMM."""
    model = RegimeHMM(cfg)
    model.fit(features_df)
    return model


if __name__ == "__main__":
    import pandas as pd
    from features import compute_features, scale_features

    cfg = Config()
    df = pd.read_parquet(".cache/synthetic_test_data.parquet")
    feats = compute_features(df, cfg)
    scaled, scaler = scale_features(feats)

    model = fit_regime_model(scaled, cfg)
    print(f"Backend used: {model.backend_}")

    states = model.predict_states(scaled)
    proba = model.predict_proba(scaled)
    print("State distribution:\n", pd.Series(states).value_counts().sort_index())
    print("Proba shape:", proba.shape)
    print("Log-likelihood:", model.score_samples(scaled))
