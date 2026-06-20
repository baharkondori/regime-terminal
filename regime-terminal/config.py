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
    asset_class: str = "crypto"   # "crypto" (24/7 trading) or "stock" (~6.5h/day, 5 days/week)
    interval: str = "1h"
    lookback_days: int = 730
    cache_dir: str = ".cache"

    # ---- HMM / regime detection ----
    n_components: int = 7
    hmm_backend: str = "hmmlearn"  # "hmmlearn" or "gmm" (sklearn fallback)
    hmm_covariance_type: str = "full"
    hmm_n_iter: int = 1000
    hmm_tol: float = 0.05
    hmm_verbose: bool = False
    random_state: int = 42
    rolling_vol_window: int = 0  # 0 = auto-derive from asset_class (see bars_per_day); set explicitly to override

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

    @property
    def bars_per_day(self) -> int:
        """How many hourly bars make up one trading day, given asset_class.
        Crypto trades 24/7; US stocks trade ~6.5h/day (9:30am-4:00pm ET),
        which rounds to 7 hourly bars. This is intentionally an approximation
        (it doesn't model half-days, holidays, or non-US exchanges) -- good
        enough for sizing rolling windows sensibly, not a market-calendar
        replacement."""
        if self.asset_class == "stock":
            return 7
        return 24  # crypto (and the default/fallback for any other class)

    @property
    def bars_per_year(self) -> int:
        """How many hourly bars make up one trading year, given asset_class.
        Crypto: 24 * 365. Stocks: ~7 bars/day * ~252 trading days/year (the
        standard US market convention, excluding weekends and holidays)."""
        if self.asset_class == "stock":
            return self.bars_per_day * 252
        return self.bars_per_day * 365

    @property
    def effective_rolling_vol_window(self) -> int:
        """The actual window used by features.py: rolling_vol_window if
        explicitly set (non-zero), otherwise one trading day's worth of
        bars for this asset_class."""
        return self.rolling_vol_window if self.rolling_vol_window > 0 else self.bars_per_day

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
