"""
utils.py
--------
Shared helpers: performance metrics (Sharpe, Sortino, max drawdown), seed
setting for reproducibility, and Plotly plotting helpers used by the
dashboard (kept here so dashboard.py stays focused on layout/controls).
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 24 * 365) -> float:
    """Annualized Sharpe ratio from a series of per-bar returns."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    return float(np.sqrt(periods_per_year) * excess.mean() / returns.std())


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 24 * 365) -> float:
    """Annualized Sortino ratio (penalizes only downside deviation)."""
    if returns.empty:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, returned as a negative fraction (e.g. -0.23)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Full drawdown series over time, for plotting."""
    running_max = equity.cummax()
    return (equity - running_max) / running_max


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def config_to_dict(cfg) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def config_from_dict(d: dict):
    from config import Config
    return Config(**d)


# ---------------------------------------------------------------------------
# Plotly helpers (used by dashboard.py)
# ---------------------------------------------------------------------------

REGIME_COLORS = {
    "crash": "#7f1d1d",
    "bear": "#b91c1c",
    "weak_bear": "#f97316",
    "chop": "#94a3b8",
    "weak_bull": "#86efac",
    "bull": "#22c55e",
    "strong_bull": "#15803d",
}


def regime_price_chart(df: pd.DataFrame, regime_names: np.ndarray):
    """Plotly figure: close price line with background shading per regime segment."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#e2e8f0", width=1.3)))

    # Shade contiguous regime runs as vrects (keeps shape count manageable)
    names = pd.Series(regime_names, index=df.index)
    change_points = names.ne(names.shift()).cumsum()
    for _, segment in names.groupby(change_points):
        start, end = segment.index[0], segment.index[-1]
        color = REGIME_COLORS.get(segment.iloc[0], "#64748b")
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.18, layer="below", line_width=0)

    fig.update_layout(
        title="Price with Regime Overlay",
        template="plotly_dark",
        height=420,
        margin=dict(l=40, r=20, t=50, b=30),
    )
    return fig


def equity_curve_chart(equity: pd.Series, benchmark: pd.Series):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Strategy", line=dict(color="#22c55e", width=2)))
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, name="Buy & Hold", line=dict(color="#64748b", width=1.5, dash="dot")))
    fig.update_layout(
        title="Equity Curve: Strategy vs Buy & Hold",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=30),
    )
    return fig


def drawdown_chart(equity: pd.Series):
    import plotly.graph_objects as go

    dd = drawdown_series(equity) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown %", line=dict(color="#ef4444")))
    fig.update_layout(
        title="Drawdown",
        template="plotly_dark",
        height=220,
        margin=dict(l=40, r=20, t=50, b=30),
        yaxis_title="%",
    )
    return fig


def posterior_heatmap(proba: np.ndarray, index: pd.Index, state_names: list):
    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Heatmap(
            z=proba.T,
            x=index,
            y=state_names,
            colorscale="Viridis",
            colorbar=dict(title="P(state)"),
        )
    )
    fig.update_layout(
        title="Regime Posterior Probabilities",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=50, b=30),
    )
    return fig
