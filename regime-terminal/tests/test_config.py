"""Unit tests for config.py's asset_class-derived properties."""

from config import Config


def test_crypto_bars_per_day():
    cfg = Config(asset_class="crypto")
    assert cfg.bars_per_day == 24


def test_stock_bars_per_day():
    cfg = Config(asset_class="stock")
    assert cfg.bars_per_day == 7


def test_crypto_bars_per_year():
    cfg = Config(asset_class="crypto")
    assert cfg.bars_per_year == 24 * 365


def test_stock_bars_per_year():
    cfg = Config(asset_class="stock")
    assert cfg.bars_per_year == 7 * 252


def test_stock_bars_per_year_less_than_crypto():
    """The core property this whole feature is about: a stock-year has
    meaningfully fewer bars than a crypto-year, since stocks don't trade
    24/7 or on weekends."""
    crypto_cfg = Config(asset_class="crypto")
    stock_cfg = Config(asset_class="stock")
    assert stock_cfg.bars_per_year < crypto_cfg.bars_per_year


def test_unknown_asset_class_falls_back_to_crypto_defaults():
    cfg = Config(asset_class="something_unrecognized")
    assert cfg.bars_per_day == 24
    assert cfg.bars_per_year == 24 * 365


def test_effective_rolling_vol_window_defaults_to_bars_per_day():
    crypto_cfg = Config(asset_class="crypto", rolling_vol_window=0)
    stock_cfg = Config(asset_class="stock", rolling_vol_window=0)
    assert crypto_cfg.effective_rolling_vol_window == 24
    assert stock_cfg.effective_rolling_vol_window == 7


def test_effective_rolling_vol_window_respects_explicit_override():
    cfg = Config(asset_class="crypto", rolling_vol_window=50)
    assert cfg.effective_rolling_vol_window == 50


def test_default_config_is_crypto():
    """The project's existing default (BTC-USD) should remain crypto
    behavior unless explicitly changed -- this guards against accidentally
    changing default behavior for all existing crypto usage."""
    cfg = Config()
    assert cfg.asset_class == "crypto"
    assert cfg.bars_per_day == 24
    assert cfg.bars_per_year == 24 * 365
