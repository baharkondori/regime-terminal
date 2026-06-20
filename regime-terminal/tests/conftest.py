"""
conftest.py
-----------
Shared pytest fixtures. Generates deterministic synthetic OHLCV data with
clearly defined regime-switching behavior so tests don't depend on network
access (yfinance) and have known ground-truth properties to assert against.
"""

import numpy as np
import pandas as pd
import pytest

from config import Config


@pytest.fixture
def synthetic_ohlcv():
    """Deterministic synthetic OHLCV data with 3 obvious regimes:
    a strong uptrend, a flat/choppy period, and a strong downtrend.
    Useful for testing that the pipeline can distinguish bull/bear/chop.
    """
    rng = np.random.RandomState(42)
    n_per_regime = 300

    segments = []
    # Strong uptrend
    segments.append(rng.normal(0.0015, 0.004, n_per_regime))
    # Choppy / flat
    segments.append(rng.normal(0.0000, 0.002, n_per_regime))
    # Strong downtrend / crash
    segments.append(rng.normal(-0.0020, 0.006, n_per_regime))

    returns = np.concatenate(segments)
    n = len(returns)
    price = 100 * np.exp(np.cumsum(returns))

    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    high = price * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.001, n)))
    openp = price * (1 + rng.normal(0, 0.0005, n))
    volume = rng.lognormal(8, 0.3, n)

    df = pd.DataFrame(
        {"Open": openp, "High": high, "Low": low, "Close": price, "Volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def small_config():
    """A Config with small/fast parameters suitable for unit tests."""
    return Config(
        n_components=3,
        hmm_n_iter=50,
        rolling_vol_window=12,
        ma_period=20,
        random_state=42,
    )
