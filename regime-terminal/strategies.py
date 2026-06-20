"""
strategies.py
-------------
Indicator confirmation layer. Computes a fixed set of (up to 8) technical
indicators and, per-bar, evaluates which ones agree with a bullish or
bearish thesis. The backtester/dashboard then requires N-of-8 confirmations
before allowing entry.

Confirmations implemented (8 total):
    1. RSI            - not overbought (bull) / not oversold (bear)
    2. Momentum (ROC)  - rate of change positive (bull) / negative (bear)
    3. ADX             - trend strength above threshold
    4. MACD            - MACD line above/below signal line
    5. Volatility      - realized vol filter (not in an extreme-vol spike)
    6. Volume spike    - volume above its rolling mean by N std devs
    7. Price action    - higher-highs/higher-lows breakout structure
    8. Moving average  - price above/below a rolling MA

Public API:
    compute_indicators(df, cfg) -> pd.DataFrame (all indicator columns)
    evaluate_confirmations(indicators_row, direction, cfg) -> dict[str, bool]
    confirmations_count(confirmations_dict) -> int
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import pandas as pd

from config import Config

Direction = Literal["bull", "bear"]

CONFIRMATION_NAMES = [
    "rsi", "momentum", "adx", "macd", "volatility", "volume_spike", "price_action", "moving_average"
]


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(periods=period) * 100


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx.fillna(0)


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute every indicator the confirmation layer needs, all in one pass."""
    out = pd.DataFrame(index=df.index)
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    out["rsi"] = _rsi(close, cfg.rsi_period)
    out["roc"] = _roc(close, cfg.roc_period)
    out["adx"] = _adx(high, low, close, cfg.adx_period)

    macd_line, signal_line = _macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out["macd_line"] = macd_line
    out["macd_signal"] = signal_line

    returns = np.log(close / close.shift(1))
    out["realized_vol"] = returns.rolling(24).std()
    out["realized_vol_pct"] = out["realized_vol"].rank(pct=True)  # percentile vs own history

    vol_mean = volume.rolling(24).mean()
    vol_std = volume.rolling(24).std()
    out["volume_z"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    rolling_high = high.rolling(20).max()
    rolling_low = low.rolling(20).min()
    out["breakout_up"] = close > rolling_high.shift(1)
    out["breakout_down"] = close < rolling_low.shift(1)

    out["ma"] = close.rolling(cfg.ma_period).mean()
    out["close"] = close

    return out


# ---------------------------------------------------------------------------
# Confirmation evaluation
# ---------------------------------------------------------------------------

def evaluate_confirmations(row: pd.Series, direction: Direction, cfg: Config) -> Dict[str, bool]:
    """Evaluate each of the 8 confirmation checks for a single bar (a row
    from compute_indicators' output), given a candidate trade direction.

    Returns a dict {confirmation_name: bool}.
    """
    confirmations: Dict[str, bool] = {}

    if direction == "bull":
        confirmations["rsi"] = bool(row["rsi"] < cfg.rsi_bull_max)
        confirmations["momentum"] = bool(row["roc"] > 0)
        confirmations["macd"] = bool(row["macd_line"] > row["macd_signal"])
        confirmations["price_action"] = bool(row["breakout_up"])
        confirmations["moving_average"] = bool(row["close"] > row["ma"]) if not pd.isna(row["ma"]) else False
    else:  # bear
        confirmations["rsi"] = bool(row["rsi"] > cfg.rsi_bear_min)
        confirmations["momentum"] = bool(row["roc"] < 0)
        confirmations["macd"] = bool(row["macd_line"] < row["macd_signal"])
        confirmations["price_action"] = bool(row["breakout_down"])
        confirmations["moving_average"] = bool(row["close"] < row["ma"]) if not pd.isna(row["ma"]) else False

    # Direction-agnostic confirmations
    confirmations["adx"] = bool(row["adx"] > cfg.adx_threshold)
    confirmations["volatility"] = bool(row["realized_vol_pct"] < 0.85) if not pd.isna(row["realized_vol_pct"]) else False
    confirmations["volume_spike"] = bool(row["volume_z"] > cfg.vol_spike_std) if not pd.isna(row["volume_z"]) else False

    return confirmations


def confirmations_count(confirmations: Dict[str, bool]) -> int:
    return sum(1 for v in confirmations.values() if v)


if __name__ == "__main__":
    import pandas as pd

    cfg = Config()
    df = pd.read_parquet(".cache/synthetic_test_data.parquet")
    indicators = compute_indicators(df, cfg)
    print(indicators.tail())

    last_row = indicators.dropna().iloc[-1]
    bull_conf = evaluate_confirmations(last_row, "bull", cfg)
    print("\nBull confirmations on last bar:", bull_conf)
    print("Count:", confirmations_count(bull_conf), "/", cfg.n_confirmations_total)
