"""
config.py
---------
Central place for all default hyperparameters and config dataclasses used
across the regime terminal. Import `Config` and override fields as needed:

    from config import Config
    cfg = Config(asset="ETH-USD", leverage=3.0)

Every other module reads parameters from a `Config` instance rather than
hardcoding values, so the whole system can be reconfigured from one place
(or from the Streamlit sidebar, or from cli.py args).
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ---- Data ----
    asset: str = "BTC-USD"
    interval: str = "1h"
    lookback_days: int = 730
    cache_dir: str = ".cache"

    # ---- HMM / regime detection ----
    n_components: int = 7
    hmm_backend: str = "hmmlearn"  # "hmmlearn" or "gmm" (sklearn fallback)
    hmm_covariance_type: str = "full"
    hmm_n_iter: int = 200
    random_state: int = 42
    rolling_vol_window: int = 24  # hours, for extra robustness feature

    # ---- Strategy / confirmations ----
    confirmations_required: int = 7
    n_confirmations_total: int = 8
    rsi_period: int = 14
    rsi_bull_max: float = 90.0   # require RSI < this for a bullish entry
    rsi_bear_min: float = 10.0   # require RSI > this for a bearish exit confirmation
    roc_period: int = 12
    adx_period: int = 14
    adx_threshold: float = 20.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vol_spike_std: float = 1.5    # volume spike threshold in std-devs
    ma_period: int = 50

    # ---- Risk management ----
    leverage: float = 2.5
    cooldown_hours: int = 48
    min_hold_hours: int = 24
    trade_cost: float = 0.0005   # 0.05% commission + slippage, per side
    stop_loss_pct: float = 0.08
    trailing_stop_pct: float = 0.04
    use_trailing_stop: bool = False

    # ---- Aggressive mode overrides ----
    aggressive_leverage: float = 4.0
    aggressive_confirmations_required: int = 5
    aggressive_use_trailing_stop: bool = True
    aggressive_trailing_stop_pct: float = 0.015

    # ---- Backtest ----
    initial_equity: float = 10_000.0

    def apply_aggressive(self) -> "Config":
        """Return a new Config with aggressive-mode overrides applied."""
        import copy
        agg = copy.deepcopy(self)
        agg.leverage = self.aggressive_leverage
        agg.confirmations_required = self.aggressive_confirmations_required
        agg.use_trailing_stop = self.aggressive_use_trailing_stop
        agg.trailing_stop_pct = self.aggressive_trailing_stop_pct
        return agg


DISCLAIMER = (
    "This software is for educational and research purposes only. "
    "Nothing produced by this codebase constitutes financial advice. "
    "Backtested performance does not guarantee future results. "
    "Cryptocurrency and leveraged trading carry substantial risk of loss. "
    "Use at your own risk and consult a licensed financial advisor before "
    "trading with real capital."
)
