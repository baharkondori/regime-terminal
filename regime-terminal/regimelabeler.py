"""
regimelabeler.py
-----------------
Takes raw HMM state IDs and auto-identifies what each one represents by
ranking states on mean forward return (and volatility as a tiebreak /
sanity check). The highest-mean-return state is labeled "bull", the
lowest is labeled "crash" (or "bear" if it's not extreme), and everything
in between is bucketed into chop/neutral-ish labels.

This mapping is stored so it can be reused consistently between retrains
(state IDs are arbitrary/unordered each time a model is refit, so without
this the labels would shuffle every time you retrain).

Public API:
    label_regimes(df, states, proba, cfg) -> (state_summary_df, mapping_dict)
    apply_labels(states, mapping_dict) -> list[str] of regime names per row
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import Config

# Regime name buckets, ordered from worst to best.
_BUCKET_NAMES_7 = [
    "crash", "bear", "weak_bear", "chop", "weak_bull", "bull", "strong_bull"
]


def _bucket_names_for_n(n: int) -> list:
    """Generate a reasonable ordered list of regime names for n_components
    that isn't necessarily 7 (keeps the labeler usable if the user changes
    n_components in the dashboard)."""
    if n == 7:
        return _BUCKET_NAMES_7
    if n <= 1:
        return ["chop"] * max(n, 1)
    names = []
    for i in range(n):
        frac = i / (n - 1)  # 0..1
        if frac < 0.15:
            names.append("crash")
        elif frac < 0.35:
            names.append("bear")
        elif frac < 0.45:
            names.append("weak_bear")
        elif frac < 0.55:
            names.append("chop")
        elif frac < 0.65:
            names.append("weak_bull")
        elif frac < 0.85:
            names.append("bull")
        else:
            names.append("strong_bull")
    return names


def label_regimes(
    returns: pd.Series,
    states: np.ndarray,
    cfg: Config,
):
    """Build a state summary table and an auto-generated state_id -> name mapping.

    Parameters
    ----------
    returns : pd.Series of per-bar returns, same length/index as `states`
    states  : np.ndarray of int state ids (output of RegimeHMM.predict_states)
    cfg     : Config

    Returns
    -------
    state_summary : DataFrame indexed by state_id with columns
                     [mean_return, volatility, count, pct_of_time, regime_name]
    mapping       : dict {state_id: regime_name}
    """
    df = pd.DataFrame({"state": states, "ret": returns.values})

    summary = df.groupby("state")["ret"].agg(
        mean_return="mean", volatility="std", count="count"
    )
    summary["pct_of_time"] = summary["count"] / summary["count"].sum()

    # Rank states by mean return, worst -> best
    ordered_states = summary.sort_values("mean_return").index.tolist()
    n = len(ordered_states)
    names = _bucket_names_for_n(n)

    mapping: Dict[int, str] = {state_id: names[i] for i, state_id in enumerate(ordered_states)}
    summary["regime_name"] = summary.index.map(mapping)

    summary = summary.sort_values("mean_return")
    return summary, mapping


def apply_labels(states: np.ndarray, mapping: Dict[int, str]) -> np.ndarray:
    """Vectorized mapping of state ids -> regime name strings."""
    return np.array([mapping.get(s, "unknown") for s in states])


BULLISH_LABELS = {"weak_bull", "bull", "strong_bull"}
BEARISH_LABELS = {"crash", "bear", "weak_bear"}
NEUTRAL_LABELS = {"chop"}


def is_bullish(regime_name: str) -> bool:
    return regime_name in BULLISH_LABELS


def is_bearish_or_crash(regime_name: str) -> bool:
    return regime_name in BEARISH_LABELS


if __name__ == "__main__":
    import pandas as pd
    from features import compute_features, scale_features
    from hmmmodel import fit_regime_model

    cfg = Config()
    df = pd.read_parquet(".cache/synthetic_test_data.parquet")
    feats = compute_features(df, cfg)
    scaled, scaler = scale_features(feats)
    model = fit_regime_model(scaled, cfg)
    states = model.predict_states(scaled)

    summary, mapping = label_regimes(feats["returns"], states, cfg)
    print(summary)
    print("\nMapping:", mapping)

    labels = apply_labels(states, mapping)
    print("\nLabel distribution:")
    print(pd.Series(labels).value_counts())
