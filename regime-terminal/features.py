"""
features.py
------------
Computes the feature set the HMM trains on, plus a few extra rolling
statistics for robustness. Also exposes a StandardScaler-based
normalization helper since HMM/GMM fitting is sensitive to feature scale.

Core HMM features (per the spec):
    - returns:        log return of Close
    - range:          (High - Low) / Close   (normalized intraday range)
    - volume_change:  pct_change of Volume

Extra robustness feature:
    - rolling_vol:    rolling std of returns over `cfg.rolling_vol_window` hours

Public API:
    compute_features(df, cfg) -> pd.DataFrame   (raw, unscaled features)
    scale_features(features_df, scaler=None) -> (scaled_df, fitted_scaler)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import Config

FEATURE_COLUMNS = ["returns", "range", "volume_change", "rolling_vol"]


def compute_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute the raw (unscaled) feature set used for HMM training.

    Parameters
    ----------
    df : DataFrame with columns Open, High, Low, Close, Volume
    cfg : Config

    Returns
    -------
    DataFrame indexed same as df, with columns FEATURE_COLUMNS, NaNs dropped.
    """
    out = pd.DataFrame(index=df.index)

    out["returns"] = np.log(df["Close"] / df["Close"].shift(1))
    out["range"] = (df["High"] - df["Low"]) / df["Close"]
    out["volume_change"] = df["Volume"].pct_change()
    out["rolling_vol"] = out["returns"].rolling(cfg.effective_rolling_vol_window).std()

    # volume_change can be +inf when Volume goes from 0 -> positive; clean that up
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna()
    return out[FEATURE_COLUMNS]


def scale_features(features_df: pd.DataFrame, scaler: StandardScaler | None = None):
    """Standardize features (zero mean, unit variance) for HMM fitting.

    If `scaler` is provided (e.g. one fitted on training data), it is reused
    (transform only) rather than refit -- important for walk-forward/live use
    so live data is scaled consistently with what the model was trained on.

    Returns
    -------
    (scaled_df, fitted_scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        values = scaler.fit_transform(features_df.values)
    else:
        values = scaler.transform(features_df.values)

    scaled_df = pd.DataFrame(values, index=features_df.index, columns=features_df.columns)
    return scaled_df, scaler


if __name__ == "__main__":
    import pandas as pd
    cfg = Config()
    df = pd.read_parquet(".cache/synthetic_test_data.parquet")
    feats = compute_features(df, cfg)
    print(feats.describe())
    scaled, scaler = scale_features(feats)
    print(scaled.describe())
