"""
dataloader.py
-------------
Fetches OHLCV data (default: hourly BTC-USD) via yfinance, handles the
multi-index columns yfinance sometimes returns, caches to disk so repeated
runs don't re-hit the network, and offers basic resampling.

Public API:
    load_ohlcv(cfg: Config, force_refresh: bool = False) -> pd.DataFrame
    resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame
"""

from __future__ import annotations

import os
import hashlib
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import Config


def _cache_path(cfg: Config) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    key = f"{cfg.asset}_{cfg.interval}_{cfg.lookback_days}"
    fname = hashlib.md5(key.encode()).hexdigest() + ".parquet"
    return os.path.join(cfg.cache_dir, fname)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns a MultiIndex on columns (ticker, field).
    Flatten to plain ['Open','High','Low','Close','Volume'] columns."""
    if isinstance(df.columns, pd.MultiIndex):
        # Typically level 0 = field, level 1 = ticker (or vice versa).
        # Find which level has the OHLCV field names.
        lvl0 = set(df.columns.get_level_values(0))
        ohlcv_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if lvl0 & ohlcv_fields:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)
    return df


def load_ohlcv(cfg: Config, force_refresh: bool = False) -> pd.DataFrame:
    """Load hourly OHLCV data for cfg.asset, using on-disk cache when possible.

    Returns a DataFrame indexed by UTC timestamp with columns:
        Open, High, Low, Close, Volume
    """
    cache_file = _cache_path(cfg)

    if not force_refresh and os.path.exists(cache_file):
        cached = pd.read_parquet(cache_file)
        # Refresh if cache is stale (older than ~ half the bar interval, capped at 1 day)
        cache_age = datetime.utcnow() - datetime.utcfromtimestamp(os.path.getmtime(cache_file))
        if cache_age < timedelta(hours=6):
            return cached

    # yfinance hourly data is limited to ~730 days lookback by the API itself.
    period_days = min(cfg.lookback_days, 729)
    ticker = yf.Ticker(cfg.asset)
    df = ticker.history(period=f"{period_days}d", interval=cfg.interval, auto_adjust=False)

    if df.empty:
        # Fall back to start/end date params if `period` fails for any reason
        end = datetime.utcnow()
        start = end - timedelta(days=period_days)
        df = ticker.history(start=start, end=end, interval=cfg.interval, auto_adjust=False)

    if df.empty:
        raise RuntimeError(
            f"No data returned for {cfg.asset} at interval={cfg.interval}. "
            "Check the ticker symbol and your network connection."
        )

    df = _flatten_columns(df)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    df.to_parquet(cache_file)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str = "4H") -> pd.DataFrame:
    """Resample an hourly OHLCV frame to a coarser timeframe, e.g. '4H', '1D'."""
    out = pd.DataFrame()
    out["Open"] = df["Open"].resample(rule).first()
    out["High"] = df["High"].resample(rule).max()
    out["Low"] = df["Low"].resample(rule).min()
    out["Close"] = df["Close"].resample(rule).last()
    out["Volume"] = df["Volume"].resample(rule).sum()
    return out.dropna()


if __name__ == "__main__":
    cfg = Config()
    data = load_ohlcv(cfg)
    print(data.tail())
    print(f"\nLoaded {len(data)} rows for {cfg.asset} from {data.index.min()} to {data.index.max()}")
