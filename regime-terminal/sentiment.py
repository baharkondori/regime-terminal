"""
sentiment.py
------------
Optional integration with LunarCrush's social sentiment data, as an
experimental additional feature for the HMM to learn from, alongside the
existing price/volume features in features.py.

Honest framing: adding a sentiment feature is a genuine experiment, not a
guaranteed improvement. Social sentiment for crypto is a real, distinct
signal from price/volume alone, but whether it actually helps THIS model
distinguish regimes better is an empirical question -- test with
walk-forward validation (see walkforward.py) comparing with vs. without
this feature before trusting it.

Data source: LunarCrush's public Topic Time Series API.
    GET https://lunarcrush.com/api4/public/topic/{topic}/time-series/v1
    Key fields used: sentiment (0-100, % of posts positive, weighted by
    engagement), galaxy_score (0-100, proprietary health/momentum score).

This module is written to work standalone with a requests-based fetch
function (fetch_sentiment_history), so it can be tested and used outside
of any specific MCP/connector setup. If you're using LunarCrush via an
MCP connector in a different context (e.g. an agent with the LunarCrush
tool available), you can skip fetch_sentiment_history and just pass
already-fetched data into compute_sentiment_features() directly --
that function only needs a DataFrame with timestamp/sentiment/galaxy_score
columns, regardless of how you obtained it.

Public API:
    fetch_sentiment_history(topic, api_key, bucket="hour", start=None, end=None) -> pd.DataFrame
    compute_sentiment_features(sentiment_df) -> pd.DataFrame (aligned, normalized features)
    merge_sentiment_with_price_features(price_features_df, sentiment_features_df) -> pd.DataFrame
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

LUNARCRUSH_BASE_URL = "https://lunarcrush.com/api4/public/topic"

# LunarCrush topics are lowercase, alphanumeric (+ spaces, #, $). Common
# crypto assets map to simple topic slugs; this is not exhaustive -- check
# LunarCrush's /public/topics/list/v1 endpoint for the full set.
ASSET_TO_TOPIC = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    "DOGE-USD": "dogecoin",
}


def asset_to_topic(asset: str) -> str:
    """Best-effort mapping from a dataloader-style asset ticker (e.g.
    'BTC-USD') to the LunarCrush topic slug it needs (e.g. 'bitcoin')."""
    if asset in ASSET_TO_TOPIC:
        return ASSET_TO_TOPIC[asset]
    # Fallback: strip the "-USD" suffix and lowercase, which works for some
    # but not all assets -- LunarCrush's topic list should be checked for
    # anything not in the explicit mapping above.
    return asset.split("-")[0].lower()


def fetch_sentiment_history(
    topic: str,
    api_key: Optional[str] = None,
    bucket: str = "hour",
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch historical sentiment/galaxy_score time-series for a topic from
    LunarCrush's public API.

    Parameters
    ----------
    topic : LunarCrush topic slug, e.g. "bitcoin" (see asset_to_topic())
    api_key : LunarCrush API key. If None, reads from the LUNARCRUSH_API_KEY
              environment variable.
    bucket : "hour" or "day" -- should match your price data's granularity
             (the rest of this project uses hourly bars by default)
    start, end : unix timestamps to bound the query. If both None, the API
                 returns its default lookback window.

    Returns
    -------
    DataFrame indexed by timestamp (UTC), with at least 'sentiment' and
    'galaxy_score' columns. Raises if the API key is missing or the request
    fails -- this is a real network call, not something to silently no-op.
    """
    import requests  # imported lazily so this module can be imported (and
    # its pure functions tested) even in environments without `requests`
    # or without ever intending to make a live call.

    key = api_key or os.environ.get("LUNARCRUSH_API_KEY")
    if not key:
        raise ValueError(
            "LunarCrush API key required. Pass api_key= explicitly, or set "
            "the LUNARCRUSH_API_KEY environment variable. Get a key at "
            "https://lunarcrush.com/developers"
        )

    url = f"{LUNARCRUSH_BASE_URL}/{topic}/time-series/v1"
    params = {"bucket": bucket}
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end

    response = requests.get(
        url, params=params, headers={"Authorization": f"Bearer {key}"}, timeout=30
    )
    response.raise_for_status()
    payload = response.json()

    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not rows:
        raise ValueError(f"LunarCrush returned no data for topic '{topic}'.")

    df = pd.DataFrame(rows)
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    elif "timestamp" not in df.columns:
        raise ValueError(
            f"Unexpected LunarCrush response shape for topic '{topic}': "
            f"no 'time' or 'timestamp' field found in {list(df.columns)}"
        )
    df = df.set_index("timestamp").sort_index()
    return df


def compute_sentiment_features(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw LunarCrush sentiment data into normalized features suitable
    for feeding into the HMM alongside the existing price/volume features.

    Parameters
    ----------
    sentiment_df : DataFrame indexed by timestamp with at least a
                   'sentiment' column (0-100 scale) and ideally a
                   'galaxy_score' column (0-100 scale). Extra columns are
                   ignored.

    Returns
    -------
    DataFrame with columns:
        sentiment_norm       : sentiment rescaled to roughly [-1, 1]
                                (50 -> 0, 100 -> 1, 0 -> -1), so "neutral"
                                sentiment doesn't bias the HMM the way a
                                raw 0-100 scale centered at 50 would.
        sentiment_change     : bar-to-bar change in sentiment_norm
        galaxy_score_norm    : galaxy_score rescaled to roughly [0, 1]
                                (only included if galaxy_score is present)
    """
    if "sentiment" not in sentiment_df.columns:
        raise ValueError("sentiment_df must contain a 'sentiment' column.")

    out = pd.DataFrame(index=sentiment_df.index)
    out["sentiment_norm"] = (sentiment_df["sentiment"] - 50) / 50
    out["sentiment_change"] = out["sentiment_norm"].diff()

    if "galaxy_score" in sentiment_df.columns:
        out["galaxy_score_norm"] = sentiment_df["galaxy_score"] / 100

    return out.dropna()


def merge_sentiment_with_price_features(
    price_features_df: pd.DataFrame,
    sentiment_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align sentiment features onto the price features' index (inner join
    on timestamp) so the combined DataFrame can be fed into
    scale_features()/fit_regime_model() exactly like the price-only case.

    Rows where sentiment data isn't available (e.g. outside LunarCrush's
    history, or a gap in their data) are dropped rather than filled, since
    fabricating sentiment values would be worse than just not having that
    bar in the (smaller) combined dataset.
    """
    combined = price_features_df.join(sentiment_features_df, how="inner")
    return combined.dropna()


if __name__ == "__main__":
    print(
        "sentiment.py requires a live LunarCrush API key to fetch real data.\n"
        "Set LUNARCRUSH_API_KEY in your environment, then run:\n\n"
        "    from sentiment import fetch_sentiment_history, asset_to_topic\n"
        "    topic = asset_to_topic('BTC-USD')\n"
        "    df = fetch_sentiment_history(topic)\n"
        "    print(df.tail())\n"
    )
