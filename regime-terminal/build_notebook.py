"""
Build script: generates notebooks/regime_terminal_colab.ipynb
Run once: python build_notebook.py
"""
import nbformat as nbf
import sys
sys.path.insert(0, '.')
from config import DISCLAIMER

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# --- Title & disclaimer ---
md("""# 📈 Regime Terminal — HMM-Driven Trading System (Colab Demo)

This notebook walks through the full pipeline:
1. Data loading (BTC-USD, hourly, last 730 days)
2. Feature engineering
3. HMM training & regime auto-labeling (7 states)
4. Strategy confirmations + backtest with default parameters
5. Dashboard-equivalent visualizations
6. Sample results + how to enable aggressive mode

> ⚠️ **Disclaimer:** This notebook is for educational and research purposes only.
> Nothing here is financial advice. Backtested performance does not guarantee
> future results. Leveraged crypto trading carries substantial risk of loss.
""")

# --- Setup ---
md("## 1. Setup\n\nClone the repo (or upload the project files) and install dependencies.")
code("""# If running in Colab, clone your GitHub repo (replace with your URL):
# !git clone https://github.com/<your-username>/regime-terminal.git
# %cd regime-terminal

!pip install -q hmmlearn yfinance pandas numpy scikit-learn plotly joblib pytest
""")

code("""import sys
sys.path.insert(0, '.')  # ensure local modules are importable

from config import Config, DISCLAIMER
from dataloader import load_ohlcv
from features import compute_features, scale_features
from hmmmodel import fit_regime_model
from regimelabeler import label_regimes, apply_labels, is_bullish
from strategies import compute_indicators, evaluate_confirmations, confirmations_count
from backtester import run_backtest
from utils import set_seed, regime_price_chart, equity_curve_chart, drawdown_chart, posterior_heatmap

import numpy as np
import pandas as pd

print(DISCLAIMER)
""")

# --- Data loading ---
md("""## 2. Data Loading & Feature Engineering

Default config: **BTC-USD**, hourly candles, **730-day** lookback.
Change `asset` below to switch markets (e.g. `ETH-USD`, `AAPL`, `SPY`).""")
code("""cfg = Config(
    asset="BTC-USD",
    lookback_days=730,
    n_components=7,
    hmm_backend="hmmlearn",   # falls back to sklearn GaussianMixture if hmmlearn isn't installed
    confirmations_required=7,
    leverage=2.5,
    min_hold_hours=24,
    cooldown_hours=48,
    random_state=42,
)
set_seed(cfg.random_state)

df = load_ohlcv(cfg)
print(f"Loaded {len(df)} hourly bars for {cfg.asset}")
print(f"Date range: {df.index.min()}  ->  {df.index.max()}")
df.tail()
""")

code("""feats = compute_features(df, cfg)
scaled_feats, scaler = scale_features(feats)
print(f"Feature matrix shape: {scaled_feats.shape}")
feats.describe()
""")

# --- HMM training ---
md("""## 3. HMM Training & Regime Auto-Labeling

We fit a Gaussian HMM with `n_components=7` on the standardized features, then
auto-label each state by ranking on mean return: the highest-mean state is
labeled `strong_bull`, the lowest is `crash`, and the rest fall in between.""")
code("""model = fit_regime_model(scaled_feats, cfg)
print(f"HMM backend used: {model.backend_}")

states = model.predict_states(scaled_feats)
proba = model.predict_proba(scaled_feats)

state_summary, mapping = label_regimes(feats["returns"], states, cfg)
regime_names = apply_labels(states, mapping)

print("\\nState summary (ranked worst -> best mean return):")
state_summary
""")

code("""# Current regime + confidence (as of the last available bar)
last_regime = regime_names[-1]
last_confidence = float(np.max(proba[-1]))
print(f"Current regime: {last_regime}")
print(f"Confidence (posterior probability): {last_confidence:.1%}")
""")

# --- Strategy + backtest ---
md("""## 4. Strategy Confirmations + Backtest (Default Parameters)

8 confirmation checks (RSI, momentum, ADX, MACD, volatility, volume spike,
price action, moving average). Default: require **7 of 8** to enter a long,
**2.5x leverage**, **48h cooldown** after exit, **24h minimum hold**,
immediate exit on regime flip to bear/crash.""")
code("""aligned_df = df.loc[feats.index]
indicators = compute_indicators(aligned_df, cfg)
common_idx = indicators.dropna().index.intersection(feats.index)

aligned_df = aligned_df.loc[common_idx]
regime_names_aligned = pd.Series(regime_names, index=feats.index).loc[common_idx].values
proba_aligned = pd.DataFrame(proba, index=feats.index).loc[common_idx].values
indicators_aligned = indicators.loc[common_idx]

conf_counts = np.array([
    confirmations_count(evaluate_confirmations(row, "bull", cfg))
    for _, row in indicators_aligned.iterrows()
])

result = run_backtest(aligned_df, regime_names_aligned, conf_counts, cfg)

print("Backtest metrics:")
for k, v in result.metrics.items():
    print(f"  {k:25s}: {v:,.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
""")

code("""print("Per-state performance breakdown:")
result.per_state_performance
""")

code("""from backtester import _make_trades_df
trades_df = _make_trades_df(result.trades)
print(f"Total trades: {len(trades_df)}")
trades_df.tail(10)
""")

# --- Visualizations ---
md("## 5. Visualizations (Dashboard-Equivalent Charts)")
code("""regime_price_chart(aligned_df, regime_names_aligned).show()
""")

code("""state_names_ordered = [mapping[s] for s in sorted(mapping.keys())]
posterior_heatmap(proba_aligned, aligned_df.index, state_names_ordered).show()
""")

code("""equity_curve_chart(result.equity_curve, result.benchmark_curve).show()
""")

code("""drawdown_chart(result.equity_curve).show()
""")

# --- Live signal ---
md("## 6. Live Signal (Current Regime + Recommended Action)")
code("""last_conf_count = int(conf_counts[-1])
bullish_now = is_bullish(last_regime)
action = (
    "LONG / HOLD" if (bullish_now and last_conf_count >= cfg.confirmations_required)
    else "EXIT / FLAT" if not bullish_now
    else "WATCH (regime ok, confirmations low)"
)

print(f"Asset:             {cfg.asset}")
print(f"As of:             {aligned_df.index[-1]}")
print(f"Regime:            {last_regime}")
print(f"Regime confidence: {last_confidence:.1%}")
print(f"Confirmations:     {last_conf_count}/{cfg.n_confirmations_total}")
print(f"Suggested action:  {action}")
""")

# --- Aggressive mode ---
md("""## 7. Aggressive Mode

Aggressive mode overrides: leverage **4x** (vs 2.5x default), confirmations
required drops to **5/8** (vs 7/8), and a tight **1.5% trailing stop** is
enabled. Use with caution — this trades more often and with more leverage.""")
code("""aggressive_cfg = cfg.apply_aggressive()
print(f"Aggressive leverage: {aggressive_cfg.leverage}x")
print(f"Aggressive confirmations required: {aggressive_cfg.confirmations_required}/8")
print(f"Trailing stop enabled: {aggressive_cfg.use_trailing_stop} at {aggressive_cfg.trailing_stop_pct:.1%}")

agg_conf_counts = np.array([
    confirmations_count(evaluate_confirmations(row, "bull", aggressive_cfg))
    for _, row in indicators_aligned.iterrows()
])
agg_result = run_backtest(aligned_df, regime_names_aligned, agg_conf_counts, aggressive_cfg)

print("\\nAggressive mode metrics:")
for k, v in agg_result.metrics.items():
    print(f"  {k:25s}: {v:,.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
""")

# --- Save model ---
md("""## 8. Save the Trained Model

Saves with a timestamp so you can version models across retrains.""")
code("""import os
from datetime import datetime

os.makedirs("models", exist_ok=True)
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_path = f"models/regime_hmm_{cfg.asset.replace('-', '')}_{timestamp}.joblib"
model.save(model_path)
print(f"Saved model to: {model_path}")
""")

# --- Dashboard ---
md("""## 9. Run the Full Interactive Dashboard

The notebook above mirrors the Streamlit dashboard's pipeline, but for the
full interactive experience (live controls, trade log export, "ask the AI
agent" box), run the dashboard:

**Locally / in a Colab terminal:**
```
streamlit run dashboard.py
```

**In Colab directly** (using a tunnel, since Colab doesn't expose ports):
```python
!pip install -q streamlit
!npm install -g localtunnel
!streamlit run dashboard.py &>/content/logs.txt &
!npx localtunnel --port 8501
```
Then open the printed `https://*.loca.lt` URL.
""")

# --- Caveats ---
md(f"""## 10. Caveats & Notes

- **Overfitting risk:** This demo fits the HMM and evaluates the backtest on
  the *same* historical window. For real evaluation, use a walk-forward
  split: fit on an earlier period, test on a later out-of-sample period, and
  repeat across multiple rolling windows. Cross-validate `n_components` and
  confirmation thresholds rather than hand-picking values that happen to look
  good on one historical run.
- **Backtest vs. live trading:** Backtests assume fills at the close price
  with a flat cost (`trade_cost`) for slippage/commission. Real execution has
  latency, partial fills, and slippage that scales with order size and
  volatility — live performance will differ from backtested performance.
- **Regime labels are descriptive, not predictive guarantees:** "bull" means
  this HMM state historically had the highest mean return in-sample. It does
  not guarantee the next bar in that state will be profitable.

{DISCLAIMER}
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open("notebooks/regime_terminal_colab.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/regime_terminal_colab.ipynb")
