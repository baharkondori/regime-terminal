"""
dashboard.py
------------
Streamlit dashboard for the regime terminal. Run with:

    streamlit run dashboard.py

Provides:
    - Sidebar controls: asset, lookback, n_components, confirmations required,
      leverage, min-hold, cooldown, aggressive toggle.
    - Price chart with regime-colored overlay.
    - Regime posterior probability heatmap.
    - Equity curve (strategy vs buy & hold) and drawdown chart.
    - Trade log table and per-state performance breakdown.
    - Current signal panel: live regime, confidence, confirmations, action.
    - "Ask the AI agent" free-text box to adjust strategy params via simple
      keyword parsing (a lightweight stand-in for an LLM-driven config editor;
      see README for how to wire this to a real Claude API call).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import Config, DISCLAIMER
from dataloader import load_ohlcv
from features import compute_features, scale_features
from hmmmodel import fit_regime_model
from regimelabeler import label_regimes, apply_labels, is_bullish
from strategies import compute_indicators, evaluate_confirmations, confirmations_count
from backtester import run_backtest
from utils import regime_price_chart, equity_curve_chart, drawdown_chart, posterior_heatmap, set_seed

st.set_page_config(page_title="Regime Terminal", layout="wide", page_icon="📈")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Regime Terminal")
st.sidebar.caption("HMM-driven regime detection & backtesting")

asset = st.sidebar.text_input("Asset", value="BTC-USD")
lookback_days = st.sidebar.slider("Lookback (days)", 90, 729, 730 if False else 729, step=10)
n_components = st.sidebar.slider("HMM states (n_components)", 3, 10, 7)
hmm_backend = st.sidebar.selectbox("HMM backend", ["hmmlearn", "gmm"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy")
confirmations_required = st.sidebar.slider("Confirmations required (of 8)", 1, 8, 7)
leverage = st.sidebar.slider("Leverage", 1.0, 5.0, 2.5, step=0.5)
min_hold_hours = st.sidebar.slider("Min hold (hours)", 1, 96, 24)
cooldown_hours = st.sidebar.slider("Cooldown after exit (hours)", 0, 96, 48)
use_trailing_stop = st.sidebar.checkbox("Use trailing stop", value=False)

st.sidebar.markdown("---")
aggressive_mode = st.sidebar.checkbox("⚡ Aggressive mode", value=False)
st.sidebar.caption("Aggressive mode overrides the sliders above: leverage 4x, confirmations 5/8, trailing stop on (1.5%).")

run_button = st.sidebar.button("🔁 Run / Retrain", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(DISCLAIMER)


# ---------------------------------------------------------------------------
# Build config from controls
# ---------------------------------------------------------------------------

cfg = Config(
    asset=asset,
    lookback_days=lookback_days,
    n_components=n_components,
    hmm_backend=hmm_backend,
    confirmations_required=confirmations_required,
    leverage=leverage,
    min_hold_hours=min_hold_hours,
    cooldown_hours=cooldown_hours,
    use_trailing_stop=use_trailing_stop,
)
if aggressive_mode:
    cfg = cfg.apply_aggressive()


# ---------------------------------------------------------------------------
# "Ask the AI agent" box (lightweight keyword-based config editor)
# ---------------------------------------------------------------------------

st.markdown("### 💬 Ask the AI agent to adjust the strategy")
agent_prompt = st.text_input(
    "e.g. \"make aggressive mode: leverage=4, confirmations=5, add trailing stop 1%\"",
    value="",
)
if agent_prompt:
    import re
    lev_match = re.search(r"leverage\s*=?\s*([\d.]+)", agent_prompt)
    conf_match = re.search(r"confirmations?\s*=?\s*(\d)", agent_prompt)
    trail_match = re.search(r"trailing stop\s*([\d.]+)\s*%", agent_prompt)
    if lev_match:
        cfg.leverage = float(lev_match.group(1))
        st.info(f"Set leverage = {cfg.leverage}")
    if conf_match:
        cfg.confirmations_required = int(conf_match.group(1))
        st.info(f"Set confirmations_required = {cfg.confirmations_required}")
    if trail_match:
        cfg.use_trailing_stop = True
        cfg.trailing_stop_pct = float(trail_match.group(1)) / 100
        st.info(f"Enabled trailing stop at {trail_match.group(1)}%")
    st.caption(
        "Note: this is simple keyword parsing, not a live LLM call. "
        "See README for wiring this box to the Anthropic API for free-form requests."
    )


# ---------------------------------------------------------------------------
# Pipeline: data -> features -> HMM -> labels -> indicators -> backtest
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_data(asset, lookback_days):
    cfg_local = Config(asset=asset, lookback_days=lookback_days)
    return load_ohlcv(cfg_local)


def run_pipeline(cfg: Config):
    set_seed(cfg.random_state)
    df = _load_data(cfg.asset, cfg.lookback_days)

    feats = compute_features(df, cfg)
    scaled, scaler = scale_features(feats)

    model = fit_regime_model(scaled, cfg)
    states = model.predict_states(scaled)
    proba = model.predict_proba(scaled)

    summary, mapping = label_regimes(feats["returns"], states, cfg)
    regime_names = apply_labels(states, mapping)

    aligned_df = df.loc[feats.index]
    indicators = compute_indicators(aligned_df, cfg)
    common_idx = indicators.dropna().index.intersection(feats.index)

    aligned_df = aligned_df.loc[common_idx]
    regime_names_aligned = pd.Series(regime_names, index=feats.index).loc[common_idx].values
    proba_aligned = pd.DataFrame(proba, index=feats.index).loc[common_idx].values
    indicators_aligned = indicators.loc[common_idx]

    conf_counts = []
    conf_breakdown = []
    for _, row in indicators_aligned.iterrows():
        c = evaluate_confirmations(row, "bull", cfg)
        conf_counts.append(confirmations_count(c))
        conf_breakdown.append(c)
    conf_counts = np.array(conf_counts)

    result = run_backtest(aligned_df, regime_names_aligned, conf_counts, cfg)

    return {
        "df": aligned_df,
        "regime_names": regime_names_aligned,
        "proba": proba_aligned,
        "state_summary": summary,
        "mapping": mapping,
        "conf_counts": conf_counts,
        "conf_breakdown": conf_breakdown[-1] if conf_breakdown else {},
        "result": result,
        "model": model,
        "indicators": indicators_aligned,
    }


# Run once on first load, and whenever the button is clicked
if "pipeline" not in st.session_state or run_button:
    with st.spinner("Loading data, fitting HMM, running backtest..."):
        try:
            st.session_state.pipeline = run_pipeline(cfg)
            st.session_state.cfg = cfg
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

pipeline = st.session_state.pipeline


# ---------------------------------------------------------------------------
# Current signal panel
# ---------------------------------------------------------------------------

st.markdown("## 📡 Current Signal")
last_regime = pipeline["regime_names"][-1]
last_proba = pipeline["proba"][-1]
last_confidence = float(np.max(last_proba))
last_conf_count = int(pipeline["conf_counts"][-1])
bullish_now = is_bullish(last_regime)
action = "🟢 LONG / HOLD" if (bullish_now and last_conf_count >= cfg.confirmations_required) else (
    "🔴 EXIT / FLAT" if not bullish_now else "🟡 WATCH (regime ok, confirmations low)"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regime", last_regime)
c2.metric("Regime confidence", f"{last_confidence:.1%}")
c3.metric("Confirmations", f"{last_conf_count} / {cfg.n_confirmations_total}")
c4.metric("Suggested action", action)

with st.expander("Why this signal? (confirmation breakdown)"):
    st.write(pipeline["conf_breakdown"])

st.markdown("---")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.markdown("## 📈 Price & Regimes")
st.plotly_chart(regime_price_chart(pipeline["df"], pipeline["regime_names"]), use_container_width=True)

state_names_ordered = [pipeline["mapping"][s] for s in sorted(pipeline["mapping"].keys())]
st.plotly_chart(
    posterior_heatmap(pipeline["proba"], pipeline["df"].index, state_names_ordered),
    use_container_width=True,
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(equity_curve_chart(pipeline["result"].equity_curve, pipeline["result"].benchmark_curve), use_container_width=True)
with col2:
    st.plotly_chart(drawdown_chart(pipeline["result"].equity_curve), use_container_width=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

st.markdown("## 📊 Performance Metrics")
metrics = pipeline["result"].metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total return", f"{metrics['total_return_pct']:.1f}%")
m2.metric("Alpha vs B&H", f"{metrics['alpha_pct']:.1f}%")
m3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
m4.metric("Sortino", f"{metrics['sortino']:.2f}")
m5.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.1f}%")

m6, m7, m8 = st.columns(3)
m6.metric("Trades", metrics["n_trades"])
m7.metric("Win rate", f"{metrics['win_rate_pct']:.1f}%")
m8.metric("Final equity", f"${metrics['final_equity']:,.0f}")


# ---------------------------------------------------------------------------
# State summary & trade log
# ---------------------------------------------------------------------------

st.markdown("## 🧭 Regime State Summary")
st.dataframe(pipeline["state_summary"], use_container_width=True)

st.markdown("## 📋 Per-State Performance")
st.dataframe(pipeline["result"].per_state_performance, use_container_width=True)

st.markdown("## 📜 Trade Log")
from backtester import _make_trades_df
trades_df = _make_trades_df(pipeline["result"].trades)
st.dataframe(trades_df, use_container_width=True)

if not trades_df.empty:
    csv = trades_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export trade log to CSV", csv, "trade_log.csv", "text/csv")

st.markdown("---")
st.caption(DISCLAIMER)
