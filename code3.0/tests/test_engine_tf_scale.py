"""Tests that bars_per_year is computed correctly for each timeframe frequency."""
import math
import numpy as np
import pandas as pd
import pytest
from trade2.backtesting.engine import run_backtest


def _make_signal_df(n: int, freq: str) -> pd.DataFrame:
    """Minimal OHLCV + signal columns for run_backtest."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    price = 2000.0 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({
        "Open":  price,
        "High":  price + 0.5,
        "Low":   price - 0.5,
        "Close": price,
        "Volume": 100.0,
        "signal_long":  0,
        "signal_short": 0,
        "exit_long": 0,
        "exit_short": 0,
        "stop_long":  price - 1.0,
        "stop_short": price + 1.0,
        "tp_long":    price + 2.0,
        "tp_short":   price - 2.0,
        "position_size_long":  1.0,
        "position_size_short": 1.0,
    }, index=ts)
    # Put one long signal so the engine doesn't short-circuit
    df.iloc[10, df.columns.get_loc("signal_long")] = 1
    return df


def _minimal_config():
    return {
        "strategy": {"name": "test", "mode": "single_tf"},
        "hmm": {"sizing_base": 1.0, "sizing_max": 1.0, "min_prob_hard": 0.0},
        "risk": {
            "atr_stop_mult": 1.0, "atr_tp_mult": 2.0, "base_allocation_frac": 0.01,
            "max_hold_bars": 0, "trailing_enabled": False, "break_even_atr_trigger": 0.0,
        },
        "costs": {
            "spread_pips": 0, "slippage_pips": 0, "commission_rt": 0.0,
            "spread_asian_mult": 1.0, "spread_vol_mult": 1.0, "spread_vol_atr_lookback": 1,
        },
        "backtest": {
            "init_cash": 10000, "risk_per_trade": 0.01, "contract_size_oz": 100,
            "use_linear_sizing": False,
        },
        "session": {"enabled": False, "allowed_hours_utc": list(range(24))},
    }


EXPECTED_BARS_PER_YEAR = {
    "1min":  252 * 24 * 60,   # 362880
    "5min":  252 * 24 * 12,   # 72576
    "15min": 252 * 24 * 4,    # 24192
    "1h":    252 * 24,         # 6048
}


@pytest.mark.parametrize("freq,expected_bpy", list(EXPECTED_BARS_PER_YEAR.items()))
def test_n_years_matches_bars_per_year(freq, expected_bpy):
    """n_years = n_bars / bars_per_year — verify bars_per_year is correct for each freq."""
    n_bars = expected_bpy  # Exactly 1 year worth of bars at correct scale
    df = _make_signal_df(n_bars, freq)
    cfg = _minimal_config()
    metrics, _ = run_backtest(df, "test_strategy", period_label="test", config=cfg, freq=freq)
    # n_years should be ~1.0 (within 1%) if bars_per_year is right
    assert abs(metrics["n_years"] - 1.0) < 0.01, (
        f"freq={freq}: expected n_years~1.0, got {metrics['n_years']:.4f} "
        f"(bars_per_year was {n_bars / metrics['n_years']:.0f}, expected {expected_bpy})"
    )
