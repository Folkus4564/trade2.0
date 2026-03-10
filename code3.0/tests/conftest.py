"""
conftest.py - Shared pytest fixtures.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _make_ohlcv(n: int, freq: str = "1h", seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    price = 1800.0
    rows  = []
    ts    = pd.date_range("2019-01-02", periods=n, freq=freq, tz="UTC")
    for _ in range(n):
        ret   = rng.normal(0, 0.003)
        close = price * (1 + ret)
        high  = close * (1 + abs(rng.normal(0, 0.001)))
        low   = close * (1 - abs(rng.normal(0, 0.001)))
        open_ = price
        price = close
        rows.append({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": rng.uniform(100, 1000)})
    return pd.DataFrame(rows, index=ts)


@pytest.fixture(scope="session")
def ohlcv_1h():
    """Synthetic 1H OHLCV DataFrame (2000 bars, ~83 trading days)."""
    return _make_ohlcv(2000, "1h")


@pytest.fixture(scope="session")
def ohlcv_5m():
    """Synthetic 5M OHLCV DataFrame (24000 bars, ~83 trading days)."""
    return _make_ohlcv(24000, "5min")


@pytest.fixture(scope="session")
def base_config():
    """Minimal config dict for testing (no file I/O)."""
    return {
        "strategy": {"name": "test_strategy", "mode": "multi_tf"},
        "data": {
            "raw_1h_csv": "data/raw/XAUUSD_1H_2019_2025.csv",
            "raw_5m_csv": "data/raw/XAUUSD_5M_2019_2025.csv",
            "missing_bar_policy": "none",
        },
        "splits": {
            "train_start": "2019-01-01",
            "train_end":   "2022-12-31",
            "val_start":   "2023-01-01",
            "val_end":     "2023-12-31",
            "test_start":  "2024-01-01",
            "test_end":    "2025-06-30",
        },
        "hmm": {
            "n_states": 3, "n_iter": 10, "random_seed": 42,
            "min_prob_hard": 0.50, "sizing_base": 0.50, "sizing_max": 1.50,
        },
        "features": {
            "hma_period": 55, "ema_period": 21, "atr_period": 14,
            "rsi_period": 14, "adx_period": 14, "dc_period": 40,
        },
        "smc": {
            "ob_validity_bars": 20, "ob_impulse_bars": 3, "ob_impulse_mult": 1.5,
            "fvg_validity_bars": 15, "swing_lookback_bars": 20,
            "require_confluence": True, "require_pin_bar": False,
        },
        "smc_5m": {
            "ob_validity_bars": 60, "ob_impulse_bars": 3, "ob_impulse_mult": 1.5,
            "fvg_validity_bars": 36, "swing_lookback_bars": 60,
            "require_confluence": True, "require_pin_bar": False,
        },
        "regime": {"persistence_bars": 3, "adx_threshold": 20.0},
        "risk": {"atr_stop_mult": 1.5, "atr_tp_mult": 3.0},
        "costs": {
            "spread_pips": 3, "slippage_pips": 1, "commission_rt": 0.0002,
            "spread_asian_mult": 2.5, "spread_vol_mult": 1.5, "spread_vol_atr_lookback": 20,
        },
        "session": {"enabled": False, "allowed_hours_utc": list(range(7, 21))},
        "acceptance": {
            "test":  {"min_annualized_return": 0.10, "min_sharpe": 1.0, "min_drawdown": -0.35,
                      "min_profit_factor": 1.2, "min_trades": 30, "min_win_rate": 0.40},
            "train": {"min_sharpe": 0.0, "min_profit_factor": 1.0, "min_trades": 50},
            "val":   {"min_sharpe": 0.5, "min_profit_factor": 1.1, "min_trades": 20},
            "walk_forward": {"min_mean_sharpe": 0.5, "min_positive_windows": 0.75},
        },
        "artefacts": {
            "root": "artefacts", "models": "artefacts/models", "backtests": "artefacts/backtests",
            "reports": "artefacts/reports", "experiments": "artefacts/experiments",
            "approved": "artefacts/approved_strategies",
        },
    }
