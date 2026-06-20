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
DESIGN_TOKENS = {
    "bg": "#0B0E14",
    "panel": "#11151F",
    "grid": "#1C2230",
    "border": "#232938",
    "text_primary": "#E8EAED",
    "text_secondary": "#8B95A8",
    "green": "#3DDC84",
    "red": "#FF5C5C",
    "amber": "#FFB454",
    "blue": "#5B9DFF",
    "font_mono": "JetBrains Mono, IBM Plex Mono, SFMono-Regular, Consolas, monospace",
    "font_body": "Inter, -apple-system, Segoe UI, sans-serif",
}

REGIME_COLORS = {
    "crash": "#FF5C5C",
    "bear": "#FF7A7A",
    "weak_bear": "#FFB454",
    "chop": "#5A6478",
    "weak_bull": "#7FE3A8",
    "bull": "#3DDC84",
    "strong_bull": "#1FB868",
}


def _base_layout(title: str, height: int) -> dict:
    """Shared Plotly layout settings so every chart in the dashboard reads
    as one cohesive terminal rather than a set of mismatched defaults."""
    t = DESIGN_TOKENS
    return dict(
        title=dict(
            text=title,
            font=dict(family=t["font_body"], size=14, color=t["text_secondary"]),
            x=0.0, xanchor="left",
        ),
        height=height,
        margin=dict(l=50, r=24, t=44, b=36),
        paper_bgcolor=t["bg"],
        plot_bgcolor=t["panel"],
        font=dict(family=t["font_mono"], size=11, color=t["text_primary"]),
        legend=dict(
            font=dict(family=t["font_body"], size=11, color=t["text_secondary"]),
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=t["grid"], zerolinecolor=t["grid"], linecolor=t["border"],
            tickfont=dict(family=t["font_mono"], size=10, color=t["text_secondary"]),
        ),
        yaxis=dict(
            gridcolor=t["grid"], zerolinecolor=t["grid"], linecolor=t["border"],
            tickfont=dict(family=t["font_mono"], size=10, color=t["text_secondary"]),
        ),
        hoverlabel=dict(
            bgcolor=t["panel"], bordercolor=t["border"],
            font=dict(family=t["font_mono"], size=11, color=t["text_primary"]),
        ),
    )


def regime_price_chart(df: pd.DataFrame, regime_names: np.ndarray):
    """Price line with crisp vertical regime-color bands and a thin top-edge
    accent per segment -- the dashboard's signature visual element."""
    import plotly.graph_objects as go

    t = DESIGN_TOKENS
    fig = go.Figure()

    names = pd.Series(regime_names, index=df.index)
    change_points = names.ne(names.shift()).cumsum()
    y_max = df["Close"].max()
    for _, segment in names.groupby(change_points):
        start, end = segment.index[0], segment.index[-1]
        color = REGIME_COLORS.get(segment.iloc[0], t["text_secondary"])
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.10, layer="below", line_width=0)
        # Thin accent line along the top edge of each regime band -- the
        # one deliberately bold, ownable visual choice for this chart.
        fig.add_shape(
            type="line", x0=start, x1=end, y0=y_max, y1=y_max,
            line=dict(color=color, width=3), layer="above",
        )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Price",
        line=dict(color=t["text_primary"], width=1.4),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra></extra>",
    ))

    layout = _base_layout("PRICE / REGIME OVERLAY", 440)
    layout["yaxis"]["tickprefix"] = "$"
    layout["yaxis"]["tickformat"] = ",.0f"
    fig.update_layout(**layout)
    fig.update_layout(showlegend=False)
    return fig


def equity_curve_chart(equity: pd.Series, benchmark: pd.Series):
    import plotly.graph_objects as go

    t = DESIGN_TOKENS
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name="STRATEGY",
        line=dict(color=t["green"], width=2),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=benchmark.index, y=benchmark.values, name="BUY & HOLD",
        line=dict(color=t["text_secondary"], width=1.4, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra></extra>",
    ))
    layout = _base_layout("EQUITY CURVE", 360)
    layout["yaxis"]["tickprefix"] = "$"
    layout["yaxis"]["tickformat"] = ",.0f"
    fig.update_layout(**layout)
    return fig


def drawdown_chart(equity: pd.Series):
    import plotly.graph_objects as go

    t = DESIGN_TOKENS
    dd = drawdown_series(equity) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="DRAWDOWN", fill="tozeroy",
        line=dict(color=t["red"], width=1.4),
        fillcolor="rgba(255,92,92,0.12)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f}%<extra></extra>",
    ))
    layout = _base_layout("DRAWDOWN", 220)
    layout["yaxis"]["ticksuffix"] = "%"
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def posterior_heatmap(proba: np.ndarray, index: pd.Index, state_names: list):
    import plotly.graph_objects as go

    t = DESIGN_TOKENS
    fig = go.Figure(
        data=go.Heatmap(
            z=proba.T,
            x=index,
            y=[s.upper() for s in state_names],
            colorscale=[
                [0.0, t["panel"]], [0.5, "#2A6E4F"], [1.0, t["green"]],
            ],
            colorbar=dict(
                title=dict(text="P", font=dict(family=t["font_mono"], size=10, color=t["text_secondary"])),
                tickfont=dict(family=t["font_mono"], size=9, color=t["text_secondary"]),
                outlinewidth=0,
            ),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y}: %{z:.0%}<extra></extra>",
        )
    )
    layout = _base_layout("REGIME POSTERIOR PROBABILITY", 280)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig
