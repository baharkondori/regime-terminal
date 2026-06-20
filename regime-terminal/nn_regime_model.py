"""
nn_regime_model.py
-------------------
A neural-network-based alternative to RegimeHMM, using PyTorch. This is a
drop-in replacement with the exact same public interface as RegimeHMM in
hmmmodel.py (.fit, .predict_states, .predict_proba, .score_samples, .save,
.load), so it can plug into the rest of the pipeline (regimelabeler.py,
strategies.py, backtester.py, dashboard.py) completely unchanged.

Approach: Deep Embedded Clustering (DEC)-style.
    1. Train a small autoencoder (4 features -> compressed embedding -> 4
       features) to learn a nonlinear representation of the data. This lets
       the model capture relationships a Gaussian HMM's linear/Gaussian
       assumptions can't.
    2. Fit a GaussianMixture on the learned embedding (not the raw features)
       to get soft cluster assignments -- this gives us state probabilities,
       matching what predict_proba() needs to return.
    3. Map cluster IDs to regime states, identical in spirit to the HMM path.

This was validated first as a numpy/sklearn prototype (autoencoder via
MLPRegressor, manually extracting bottleneck-layer activations) before being
translated to PyTorch here, specifically so the clustering logic itself was
confirmed sound before committing to the PyTorch translation.

Honest framing, consistent with the rest of this project: this is a
different way to do the *same* unsupervised classification job the HMM
does (assign each bar to a regime cluster). It is not inherently more
"accurate" or more predictive -- it can capture more complex patterns in
principle, but it's also easier to overfit, and which one produces more
useful regime labels for your data is an empirical question worth testing,
not something to assume in advance.

Public API:
    class RegimeNN
        .fit(features_df)
        .predict_states(features_df) -> np.ndarray[int]
        .predict_proba(features_df) -> np.ndarray[T, n_components]
        .score_samples(features_df) -> float  (negative reconstruction error,
            so "higher is better" stays consistent with RegimeHMM's
            log-likelihood convention)
        .save(path) / RegimeNN.load(path)

Requires: torch (CPU build is sufficient; no GPU needed for this model size).
    pip install torch
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib

from config import Config

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.mixture import GaussianMixture


class _Autoencoder(nn.Module if _TORCH_AVAILABLE else object):
    """Small symmetric autoencoder: n_features -> hidden -> bottleneck -> hidden -> n_features.

    Architecture is intentionally small (this is 4 input features on
    ~17k rows, not image/text data) -- a large network here would just
    overfit noise in financial returns data.
    """

    def __init__(self, n_features: int, hidden_dim: int = 8, bottleneck_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


@dataclass
class RegimeNN:
    cfg: Config
    backend_: str = ""
    model_: object = None
    n_components: int = 7

    def __init__(self, cfg: Config):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for RegimeNN but is not installed in this "
                "environment. Install it with: pip install torch\n"
                "(CPU-only build is sufficient -- no GPU required for this model size.)"
            )
        self.cfg = cfg
        self.n_components = cfg.n_components
        self.autoencoder_ = None
        self.gmm_ = None
        self.backend_ = "pytorch_dec"
        self._hidden_dim = getattr(cfg, "nn_hidden_dim", 8)
        self._bottleneck_dim = getattr(cfg, "nn_bottleneck_dim", 2)
        self._epochs = getattr(cfg, "nn_epochs", 200)
        self._lr = getattr(cfg, "nn_learning_rate", 1e-2)

    def fit(self, features_df: pd.DataFrame) -> "RegimeNN":
        X = features_df.values.astype(np.float32)
        n_features = X.shape[1]

        torch.manual_seed(self.cfg.random_state)

        self.autoencoder_ = _Autoencoder(
            n_features=n_features,
            hidden_dim=self._hidden_dim,
            bottleneck_dim=self._bottleneck_dim,
        )

        X_tensor = torch.from_numpy(X)
        optimizer = torch.optim.Adam(self.autoencoder_.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        self.autoencoder_.train()
        for _ in range(self._epochs):
            optimizer.zero_grad()
            _, reconstruction = self.autoencoder_(X_tensor)
            loss = loss_fn(reconstruction, X_tensor)
            loss.backward()
            optimizer.step()

        self.autoencoder_.eval()
        with torch.no_grad():
            embedding, _ = self.autoencoder_(X_tensor)
        embedding_np = embedding.numpy()

        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.cfg.hmm_covariance_type
            if self.cfg.hmm_covariance_type in ("full", "tied", "diag", "spherical")
            else "full",
            n_init=3,
            random_state=self.cfg.random_state,
        )
        self.gmm_.fit(embedding_np)

        return self

    def _embed(self, features_df: pd.DataFrame) -> np.ndarray:
        X = features_df.values.astype(np.float32)
        X_tensor = torch.from_numpy(X)
        self.autoencoder_.eval()
        with torch.no_grad():
            embedding, _ = self.autoencoder_(X_tensor)
        return embedding.numpy()

    def predict_states(self, features_df: pd.DataFrame) -> np.ndarray:
        embedding = self._embed(features_df)
        return self.gmm_.predict(embedding)

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        embedding = self._embed(features_df)
        return self.gmm_.predict_proba(embedding)

    def score_samples(self, features_df: pd.DataFrame):
        """Returns the GMM's log-likelihood of the data under the fitted
        clustering (on the learned embedding), for consistency with
        RegimeHMM.score_samples()'s "higher is better" convention."""
        embedding = self._embed(features_df)
        return self.gmm_.score(embedding) * len(embedding)

    def reconstruction_error(self, features_df: pd.DataFrame) -> float:
        """Extra diagnostic specific to the autoencoder approach: mean squared
        reconstruction error. Useful for sanity-checking that the autoencoder
        actually learned something (very high error suggests underfitting;
        near-zero on training data alone doesn't confirm it generalizes)."""
        X = features_df.values.astype(np.float32)
        X_tensor = torch.from_numpy(X)
        self.autoencoder_.eval()
        with torch.no_grad():
            _, reconstruction = self.autoencoder_(X_tensor)
        return float(torch.mean((reconstruction - X_tensor) ** 2).item())

    def save(self, path: str) -> None:
        # Save the GMM and config via joblib, and the PyTorch weights separately
        # via torch's own serialization (joblib doesn't reliably pickle torch
        # tensors across versions).
        state = {
            "cfg": self.cfg,
            "n_components": self.n_components,
            "backend_": self.backend_,
            "gmm_": self.gmm_,
            "hidden_dim": self._hidden_dim,
            "bottleneck_dim": self._bottleneck_dim,
            "autoencoder_state_dict": self.autoencoder_.state_dict(),
            "n_features": self.autoencoder_.encoder[0].in_features,
        }
        joblib.dump(state, path)

    @staticmethod
    def load(path: str) -> "RegimeNN":
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load a RegimeNN model.")
        state = joblib.load(path)
        model = RegimeNN(state["cfg"])
        model.n_components = state["n_components"]
        model.backend_ = state["backend_"]
        model.gmm_ = state["gmm_"]
        model._hidden_dim = state["hidden_dim"]
        model._bottleneck_dim = state["bottleneck_dim"]
        model.autoencoder_ = _Autoencoder(
            n_features=state["n_features"],
            hidden_dim=state["hidden_dim"],
            bottleneck_dim=state["bottleneck_dim"],
        )
        model.autoencoder_.load_state_dict(state["autoencoder_state_dict"])
        model.autoencoder_.eval()
        return model


def fit_regime_nn(features_df: pd.DataFrame, cfg: Config) -> RegimeNN:
    """Convenience wrapper: instantiate + fit a RegimeNN, mirroring
    hmmmodel.fit_regime_model()'s convenience function."""
    model = RegimeNN(cfg)
    model.fit(features_df)
    return model


if __name__ == "__main__":
    import pandas as pd
    from features import compute_features, scale_features

    if not _TORCH_AVAILABLE:
        print(
            "PyTorch is not installed in this environment. Install it with:\n"
            "  pip install torch\n"
            "(CPU-only build is sufficient.)"
        )
    else:
        cfg = Config(n_components=4)
        df = pd.read_parquet(".cache/synthetic_test_data.parquet")
        feats = compute_features(df, cfg)
        scaled, scaler = scale_features(feats)

        model = fit_regime_nn(scaled, cfg)
        print(f"Backend used: {model.backend_}")

        states = model.predict_states(scaled)
        proba = model.predict_proba(scaled)
        print("State distribution:\n", pd.Series(states).value_counts().sort_index())
        print("Proba shape:", proba.shape)
        print("Reconstruction error:", model.reconstruction_error(scaled))
        print("Score (log-likelihood-like):", model.score_samples(scaled))
