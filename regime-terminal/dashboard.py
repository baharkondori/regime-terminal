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
from regimelabeler import label_regimes, apply_labels, is_bullish, compute_transition_table, most_likely_next_regimes
from prediction_log import log_snapshot, grade_log, summarize_accuracy, load_log
from strategies import compute_indicators, evaluate_confirmations, confirmations_count_series
from backtester import run_backtest
from utils import regime_price_chart, equity_curve_chart, drawdown_chart, posterior_heatmap, set_seed

st.set_page_config(page_title="Regime Terminal", layout="wide", page_icon="📈")

SIGNAL_LOG_PATH = "signal_log.csv"
DEFAULT_GRADING_HORIZON_BARS = 24  # how many bars ahead to check "did the historical pattern's top guess come true"

# ---------------------------------------------------------------------------
# Visual theme: trading-terminal aesthetic (dark, monospace data, sharp
# edges, signal-color coding). Tokens match utils.py's DESIGN_TOKENS so the
# Plotly charts and the surrounding Streamlit shell read as one system.
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --rt-bg: #0B0E14;
    --rt-panel: #11151F;
    --rt-border: #232938;
    --rt-text: #E8EAED;
    --rt-text-dim: #8B95A8;
    --rt-green: #3DDC84;
    --rt-red: #FF5C5C;
    --rt-amber: #FFB454;
    --rt-blue: #5B9DFF;
    --rt-mono: 'JetBrains Mono', 'IBM Plex Mono', SFMono-Regular, Consolas, monospace;
    --rt-body: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}

/* App shell */
.stApp {
    background-color: var(--rt-bg);
    font-family: var(--rt-body);
}
[data-testid="stSidebar"] {
    background-color: var(--rt-panel);
    border-right: 1px solid var(--rt-border);
}
[data-testid="stSidebar"] * {
    font-family: var(--rt-body);
}

/* Headings */
h1, h2, h3 {
    font-family: var(--rt-body) !important;
    font-weight: 600 !important;
    color: var(--rt-text) !important;
    letter-spacing: -0.01em;
}
h1 { border-bottom: 1px solid var(--rt-border); padding-bottom: 0.5rem; }

/* Metric cards: sharp edges, hairline border, monospace value */
[data-testid="stMetric"] {
    background-color: var(--rt-panel);
    border: 1px solid var(--rt-border);
    border-radius: 3px;
    padding: 0.85rem 1rem;
}
[data-testid="stMetricLabel"] {
    font-family: var(--rt-body) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: var(--rt-text-dim) !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    font-family: var(--rt-mono) !important;
    font-weight: 600 !important;
    color: var(--rt-text) !important;
}

/* Body text */
p, span, label, .stMarkdown, .stCaption {
    font-family: var(--rt-body);
    color: var(--rt-text);
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--rt-text-dim) !important;
}

/* Numeric / monospace content: dataframes, code blocks */
[data-testid="stDataFrame"], .stDataFrame, code {
    font-family: var(--rt-mono) !important;
}

/* Buttons: sharp edges, signal-green accent on primary */
.stButton > button, .stDownloadButton > button {
    border-radius: 3px;
    border: 1px solid var(--rt-border);
    font-family: var(--rt-body);
    font-weight: 500;
}
.stButton > button[kind="primary"] {
    background-color: var(--rt-green);
    border-color: var(--rt-green);
    color: #06120A;
}

/* Inputs: match panel surface */
.stTextInput input, .stSelectbox [data-baseweb="select"], .stSlider {
    font-family: var(--rt-mono);
}

/* Expander headers */
[data-testid="stExpander"] {
    border: 1px solid var(--rt-border);
    border-radius: 3px;
}

/* Dividers */
hr {
    border-color: var(--rt-border);
}

/* Info/warning boxes: align with panel surface instead of Streamlit's defaults */
[data-testid="stAlert"] {
    border-radius: 3px;
    font-family: var(--rt-body);
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Regime Terminal")
st.sidebar.caption("HMM-driven regime detection & backtesting")

# --- Asset class picker: two sections, Crypto and Stock ---
COMMON_CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"]
COMMON_STOCK_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY"]

asset_class = st.sidebar.radio("Market", ["Crypto", "Stock"], horizontal=True)

if asset_class == "Crypto":
    ticker_choice = st.sidebar.selectbox("Crypto asset", COMMON_CRYPTO_TICKERS + ["Custom..."])
    default_asset = ticker_choice if ticker_choice != "Custom..." else "BTC-USD"
    cfg_asset_class = "crypto"
else:
    ticker_choice = st.sidebar.selectbox("Stock ticker", COMMON_STOCK_TICKERS + ["Custom..."])
    default_asset = ticker_choice if ticker_choice != "Custom..." else "AAPL"
    cfg_asset_class = "stock"

if ticker_choice == "Custom...":
    asset = st.sidebar.text_input("Enter ticker", value=default_asset)
else:
    asset = ticker_choice

if cfg_asset_class == "stock":
    st.sidebar.caption(
        "Stocks trade ~6.5h/day, 5 days/week (unlike crypto's 24/7). "
        "Rolling windows and Sharpe/Sortino are automatically scaled for this "
        "(see Config.bars_per_day) rather than using crypto's 24-bars-per-day assumption."
    )

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
    asset_class=cfg_asset_class,
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
# Terminal-style header bar
# ---------------------------------------------------------------------------

import datetime as _dt
_now_utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
_class_label = "CRYPTO · 24/7" if cfg.asset_class == "crypto" else "EQUITY · MKT HRS"

st.markdown(f"""
<div style="
    display:flex; justify-content:space-between; align-items:baseline;
    border-bottom:1px solid #232938; padding-bottom:10px; margin-bottom:18px;
">
  <div style="font-family:'Inter',sans-serif; font-weight:700; font-size:1.3rem; color:#E8EAED; letter-spacing:-0.01em;">
    REGIME TERMINAL
  </div>
  <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#8B95A8; display:flex; gap:18px;">
    <span style="color:#5B9DFF; font-weight:600;">{cfg.asset}</span>
    <span>{_class_label}</span>
    <span>{_now_utc}</span>
  </div>
</div>
""", unsafe_allow_html=True)


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

    conf_counts = confirmations_count_series(indicators_aligned, "bull", cfg)
    # Breakdown dict (which specific checks passed) is only needed for the
    # single most recent bar, shown in the "why this signal" panel — so this
    # one row still uses the readable per-check function, not the full series.
    conf_breakdown = evaluate_confirmations(indicators_aligned.iloc[-1], "bull", cfg) if len(indicators_aligned) else {}

    result = run_backtest(aligned_df, regime_names_aligned, conf_counts, cfg)
    transition_table = compute_transition_table(regime_names_aligned)

    return {
        "df": aligned_df,
        "regime_names": regime_names_aligned,
        "proba": proba_aligned,
        "state_summary": summary,
        "mapping": mapping,
        "conf_counts": conf_counts,
        "conf_breakdown": conf_breakdown,
        "result": result,
        "model": model,
        "indicators": indicators_aligned,
        "transition_table": transition_table,
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

# --- Auto-log this snapshot, and grade any past snapshots whose outcome is now knowable ---
_last_regime_for_log = pipeline["regime_names"][-1]
_last_price_for_log = float(pipeline["df"]["Close"].iloc[-1])
_last_bar_ts_for_log = pipeline["df"].index[-1]
_top_next = most_likely_next_regimes(pipeline["transition_table"], _last_regime_for_log, top_n=1)
_predicted_regime, _predicted_prob = (_top_next[0] if _top_next else (None, None))

_last_conf_count_for_log = int(pipeline["conf_counts"][-1])
_action_for_log = "LONG/HOLD" if (is_bullish(_last_regime_for_log) and _last_conf_count_for_log >= cfg.confirmations_required) else (
    "EXIT/FLAT" if not is_bullish(_last_regime_for_log) else "WATCH"
)

log_snapshot(
    path=SIGNAL_LOG_PATH,
    asset=cfg.asset,
    bar_timestamp=_last_bar_ts_for_log,
    regime=_last_regime_for_log,
    confidence=float(np.max(pipeline["proba"][-1])),
    conf_count=_last_conf_count_for_log,
    conf_required=cfg.confirmations_required,
    action=_action_for_log,
    price_at_log=_last_price_for_log,
    predicted_next_regime=_predicted_regime,
    predicted_next_prob=_predicted_prob,
    horizon_bars=DEFAULT_GRADING_HORIZON_BARS,
)
graded_log = grade_log(SIGNAL_LOG_PATH, pipeline["df"], pipeline["regime_names"])


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

# --- Setup strength tier: one transparent summary of the signals above ---
from regimelabeler import compute_setup_strength
from explain import explain_setup_strength

_top_next_for_strength = most_likely_next_regimes(pipeline["transition_table"], last_regime, top_n=1)
# "Historical continuation" = probability that the historically most-likely
# next regime is itself bullish (i.e. the bullish move tends to continue).
# None if there's no transition history yet for this regime.
if _top_next_for_strength:
    _next_regime_name, _next_regime_prob = _top_next_for_strength[0]
    _historical_continuation = _next_regime_prob if is_bullish(_next_regime_name) else (1 - _next_regime_prob)
else:
    _historical_continuation = None

setup_strength = compute_setup_strength(
    regime_name=last_regime,
    regime_confidence=last_confidence,
    conf_count=last_conf_count,
    conf_required=cfg.confirmations_required,
    conf_total=cfg.n_confirmations_total,
    historical_continuation_prob=_historical_continuation,
)

_tier_colors = {"A": "#3DDC84", "B": "#FFB454", "C": "#FF8A5C", "D": "#FF5C5C"}
_tier_color = _tier_colors.get(setup_strength["tier"], "#8B95A8")
st.markdown(f"""
<div style="
    display:inline-flex; align-items:center; gap:10px;
    border:1px solid {_tier_color}; border-left:4px solid {_tier_color};
    border-radius:3px; padding:8px 16px; margin:6px 0 4px 0;
    background-color:#11151F;
">
  <span style="font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.05rem; color:{_tier_color};">
    TIER {setup_strength['tier']}
  </span>
  <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#8B95A8;">
    {setup_strength['score']}/100
  </span>
  <span style="font-family:'Inter',sans-serif; font-size:0.78rem; color:#8B95A8;">
    SETUP STRENGTH
  </span>
</div>
""", unsafe_allow_html=True)
st.caption(explain_setup_strength(setup_strength, last_regime))

# --- Plain-English explanation (for beginners) ---
from explain import explain_signal, explain_confirmation_breakdown, GLOSSARY, DISCLAIMER_SHORT, explain_stop_levels

st.info(
    explain_signal(
        regime=last_regime,
        confidence=last_confidence,
        conf_count=last_conf_count,
        conf_required=cfg.confirmations_required,
        conf_total=cfg.n_confirmations_total,
        bullish_now=bullish_now,
        action=action,
    )
)
st.caption(DISCLAIMER_SHORT)

# --- Stop-loss / trailing-stop levels for THIS moment, in real price terms ---
last_price = float(pipeline["df"]["Close"].iloc[-1])
stop_loss_price = last_price * (1 - cfg.stop_loss_pct)
trailing_stop_price = last_price * (1 - cfg.trailing_stop_pct) if cfg.use_trailing_stop else None

st.markdown("### 🛟 If you bought right now, at what price would you automatically be out?")
sl1, sl2, sl3 = st.columns(3)
sl1.metric("Current price", f"${last_price:,.2f}")
sl2.metric(f"Stop-loss ({cfg.stop_loss_pct:.0%} below)", f"${stop_loss_price:,.2f}")
if trailing_stop_price is not None:
    sl3.metric(f"Trailing stop ({cfg.trailing_stop_pct:.0%} below peak)", f"${trailing_stop_price:,.2f} *")
else:
    sl3.metric("Trailing stop", "Off")

st.caption(
    explain_stop_levels(
        last_price=last_price,
        stop_loss_price=stop_loss_price,
        stop_loss_pct=cfg.stop_loss_pct,
        trailing_stop_price=trailing_stop_price,
        trailing_stop_pct=cfg.trailing_stop_pct,
        use_trailing_stop=cfg.use_trailing_stop,
    )
)

with st.expander("Why this signal? (confirmation breakdown)"):
    for line in explain_confirmation_breakdown(pipeline["conf_breakdown"]):
        st.markdown(f"- {line}")
    st.markdown("**Raw values:**")
    st.write(pipeline["conf_breakdown"])

with st.expander("📖 What do these terms mean?"):
    for term, definition in GLOSSARY.items():
        st.markdown(f"**{term.capitalize()}** — {definition}")

# --- Historical transition pattern (what tended to happen next, historically) ---
st.markdown("### 🔄 What tended to happen next, historically")
from explain import explain_transition

transition_table = pipeline["transition_table"]
if not transition_table.empty and last_regime in transition_table.index:
    n_obs = int(transition_table.loc[last_regime, "n_observations"])
    top_next = most_likely_next_regimes(transition_table, last_regime, top_n=3)
    st.warning(explain_transition(last_regime, top_next, n_obs))

    with st.expander("Full historical transition table (every regime → every regime)"):
        st.caption(
            "Each row is a starting regime; each column is what the *next* bar turned out "
            "to be, as a historical percentage. 'n_observations' is how many times that "
            "starting regime occurred — rows with very few observations are less reliable."
        )
        display_table = transition_table.copy()
        pct_cols = [c for c in display_table.columns if c != "n_observations"]
        display_table[pct_cols] = (display_table[pct_cols] * 100).round(1)
        st.dataframe(display_table, use_container_width=True)
else:
    st.caption("Not enough historical data yet to show transition patterns for this regime.")

# --- Track record: was the historical pattern's top guess actually right, looking back? ---
st.markdown("### 📓 Track record: was the historical pattern right last time?")
st.caption(
    f"Every time you click Run/Retrain, this app saves a snapshot of the signal and its "
    f"top historical guess. Once {DEFAULT_GRADING_HORIZON_BARS} bars have passed, it checks "
    f"whether that guess actually came true and marks it correct or incorrect below — so you "
    f"can see, over time, how reliable these historical patterns have actually been for "
    f"{cfg.asset}, instead of just trusting the most recent one."
)

accuracy = summarize_accuracy(graded_log)
if accuracy["n_graded"] == 0:
    st.info(
        "No graded predictions yet — this builds up over time as you keep using the "
        f"dashboard. Each snapshot needs {DEFAULT_GRADING_HORIZON_BARS} bars to pass before "
        "it can be graded."
    )
else:
    t1, t2, t3 = st.columns(3)
    t1.metric("Predictions graded so far", accuracy["n_graded"])
    t2.metric("Correct", f"{accuracy['n_correct']} / {accuracy['n_graded']}")
    t3.metric("Accuracy", f"{accuracy['accuracy_pct']:.0f}%")
    st.caption(
        f"{accuracy['n_pending']} more snapshot(s) logged but not yet old enough to grade."
    )

with st.expander("View full signal log (all snapshots, graded and pending)"):
    if graded_log.empty:
        st.caption("No snapshots logged yet.")
    else:
        display_log = graded_log.sort_values("logged_at", ascending=False).copy()
        st.dataframe(display_log, use_container_width=True)
        csv = display_log.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export signal log to CSV", csv, "signal_log_export.csv", "text/csv")

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
