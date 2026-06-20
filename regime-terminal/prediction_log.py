"""
prediction_log.py
------------------
Logs a snapshot of the signal every time the dashboard runs/retrains, and
later grades each past snapshot against what *actually* happened — i.e.
"the historical pattern said weak_bull most often leads to mild upward
trend; did it actually go there this time, N bars later?"

This is intentionally a *hindsight* scoring system, not a live prediction
tracker: a snapshot can't be graded until enough bars have passed to see
the outcome, so grading happens lazily whenever grade_log() is called
(e.g. each time the dashboard loads), and only for rows old enough to
have a known outcome.

Storage: a single CSV file (default: signal_log.csv) in the project
directory, append-only except for the grading columns getting filled in
once an outcome is known.

Public API:
    log_snapshot(path, snapshot_dict) -> None
    grade_log(path, current_regime_names, current_index, horizon_bars=24) -> pd.DataFrame
    load_log(path) -> pd.DataFrame
    summarize_accuracy(graded_df) -> dict
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

LOG_COLUMNS = [
    "logged_at",          # when this snapshot was saved (wall-clock time)
    "bar_timestamp",      # the timestamp of the most recent bar at log time
    "asset",
    "regime",
    "confidence",
    "conf_count",
    "conf_required",
    "action",
    "price_at_log",
    "predicted_next_regime",   # top historical "most likely next" regime
    "predicted_next_prob",     # its historical probability
    "horizon_bars",            # how many bars ahead we'll check the outcome
    "outcome_known",           # bool: has enough time passed to grade this?
    "actual_regime_at_horizon",
    "prediction_correct",      # bool or None until graded
]


def _ensure_log_exists(path: str) -> None:
    if not os.path.exists(path):
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(path, index=False)


def log_snapshot(
    path: str,
    asset: str,
    bar_timestamp,
    regime: str,
    confidence: float,
    conf_count: int,
    conf_required: int,
    action: str,
    price_at_log: float,
    predicted_next_regime,
    predicted_next_prob,
    horizon_bars: int = 24,
) -> None:
    """Append one snapshot row to the log. Called once per dashboard run.

    `predicted_next_regime`/`predicted_next_prob` should be the top entry
    from regimelabeler.most_likely_next_regimes() — i.e. what historically
    tended to follow this regime most often. Pass None for both if there
    wasn't enough history to make that call.
    """
    _ensure_log_exists(path)

    # Avoid duplicate rows if the same bar gets logged twice in a row
    # (e.g. clicking Run/Retrain twice without new data arriving).
    existing = load_log(path)
    if not existing.empty:
        same_bar = existing[
            (existing["asset"] == asset) & (existing["bar_timestamp"] == str(bar_timestamp))
        ]
        if not same_bar.empty:
            return

    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "bar_timestamp": str(bar_timestamp),
        "asset": asset,
        "regime": regime,
        "confidence": confidence,
        "conf_count": conf_count,
        "conf_required": conf_required,
        "action": action,
        "price_at_log": price_at_log,
        "predicted_next_regime": predicted_next_regime,
        "predicted_next_prob": predicted_next_prob,
        "horizon_bars": horizon_bars,
        "outcome_known": False,
        "actual_regime_at_horizon": None,
        "prediction_correct": None,
    }

    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)


def load_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(path)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def grade_log(path: str, df_with_regimes: pd.DataFrame, regime_col_values: np.ndarray) -> pd.DataFrame:
    """Grade every ungraded row whose outcome is now knowable.

    Parameters
    ----------
    path : path to the log CSV
    df_with_regimes : the current pipeline's aligned OHLCV dataframe (its
                       .index is used to locate each logged bar_timestamp)
    regime_col_values : the current pipeline's regime_names array, same
                         length/order as df_with_regimes.index

    For each ungraded row: find the bar_timestamp in the current data, look
    `horizon_bars` rows ahead, and check whether the regime at that point
    matches predicted_next_regime. Rows whose horizon hasn't arrived yet in
    the available data are left ungraded.

    Returns the full (now partially graded) log DataFrame, and also
    rewrites the CSV with the updates.
    """
    log = load_log(path)
    if log.empty:
        return log

    # Ensure these columns can hold strings/bools/None regardless of how
    # pandas inferred their dtype from the CSV (an all-empty column reads
    # back as float64, which then rejects string assignment).
    for col in ("outcome_known", "actual_regime_at_horizon", "prediction_correct"):
        log[col] = log[col].astype(object)

    index_list = list(df_with_regimes.index.astype(str))
    index_pos = {ts: i for i, ts in enumerate(index_list)}

    changed = False
    for i, row in log.iterrows():
        if bool(row.get("outcome_known", False)):
            continue
        if pd.isna(row.get("predicted_next_regime")) or row.get("predicted_next_regime") in (None, "", "nan"):
            continue

        bar_ts = str(row["bar_timestamp"])
        if bar_ts not in index_pos:
            continue  # this bar isn't in the currently loaded data window

        start_pos = index_pos[bar_ts]
        horizon = int(row["horizon_bars"]) if not pd.isna(row["horizon_bars"]) else 24
        target_pos = start_pos + horizon

        if target_pos >= len(regime_col_values):
            continue  # not enough future data yet to know the outcome

        actual_regime = regime_col_values[target_pos]
        predicted = row["predicted_next_regime"]
        correct = bool(actual_regime == predicted)

        log.at[i, "outcome_known"] = True
        log.at[i, "actual_regime_at_horizon"] = actual_regime
        log.at[i, "prediction_correct"] = correct
        changed = True

    if changed:
        log.to_csv(path, index=False)

    return log


def summarize_accuracy(graded_df: pd.DataFrame) -> dict:
    """Summarize how often the historical 'most likely next regime' guess
    actually turned out to be correct, among graded rows only."""
    graded = graded_df[graded_df["outcome_known"] == True]  # noqa: E712
    n_graded = len(graded)
    if n_graded == 0:
        return {"n_graded": 0, "n_pending": len(graded_df), "accuracy_pct": None}

    n_correct = int(graded["prediction_correct"].sum())
    n_pending = len(graded_df) - n_graded

    return {
        "n_graded": n_graded,
        "n_pending": n_pending,
        "n_correct": n_correct,
        "accuracy_pct": 100.0 * n_correct / n_graded,
    }
