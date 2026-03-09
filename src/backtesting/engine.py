"""
Module: engine.py
Purpose: vectorbt-based backtesting engine for XAUUSD systematic strategy
Author: Strategy Code Engineer Agent
Date: 2026-03-08
"""

import json
import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.backtesting.metrics import compute_metrics, format_report, verdict, passes_criteria

# ── Execution cost constants ───────────────────────────────────────────────────
SPREAD_PIPS   = 3           # 3 pips spread (XAUUSD tick = $0.01)
SLIPPAGE_PIPS = 1           # 1 pip per side slippage
COMMISSION_RT = 0.0002      # 2 bps round-trip commission

BACKTESTS_DIR = Path(__file__).parents[2] / "backtests"
BACKTESTS_DIR.mkdir(exist_ok=True)


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    period_label: str = "test",
    init_cash: float = 100_000.0,
    size_pct: float  = 0.95,
) -> Tuple[Dict[str, Any], vbt.Portfolio]:
    """
    Run vectorbt backtest using signal columns already in df.

    Expects these columns in df:
    - signal_long, signal_short  (1/0)
    - exit_long, exit_short      (1/0)

    Args:
        df:             Feature+signal DataFrame
        strategy_name:  Name for saving results
        period_label:   'train', 'val', or 'test'
        init_cash:      Starting capital
        size_pct:       Fraction of portfolio per trade

    Returns:
        (metrics_dict, vbt.Portfolio)
    """
    close = df["Close"].astype(float)

    # ── Signals ───────────────────────────────────────────────────────────────
    long_entries  = df["signal_long"].astype(bool)
    long_exits    = df["exit_long"].astype(bool)
    short_entries = df["signal_short"].astype(bool)
    short_exits   = df["exit_short"].astype(bool)

    # Ensure no simultaneous long and short entries
    conflict = long_entries & short_entries
    long_entries  = long_entries  & ~conflict
    short_entries = short_entries & ~conflict

    if long_entries.sum() == 0 and short_entries.sum() == 0:
        print(f"[engine] WARNING: No signals generated for {period_label}")
        empty_metrics = {
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
        return empty_metrics, None

    # ── Slippage per-unit (convert pips to price ratio) ──────────────────────
    avg_price  = close.mean()
    pip_value  = 0.01  # XAUUSD 1 pip = $0.01
    slippage   = (SLIPPAGE_PIPS * pip_value) / avg_price

    # ── Run portfolio ─────────────────────────────────────────────────────────
    # Use fixed value sizing (init_cash * size_pct per trade) instead of
    # "percent" which does not support position reversal in vectorbt.
    trade_value = init_cash * size_pct

    pf = vbt.Portfolio.from_signals(
        close         = close,
        entries       = long_entries,
        exits         = long_exits,
        short_entries = short_entries,
        short_exits   = short_exits,
        init_cash     = init_cash,
        fees          = COMMISSION_RT / 2,   # per side
        slippage      = slippage,
        size          = trade_value,
        size_type     = "value",
        freq          = "1h",
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    equity = pf.value()
    trades_df = pf.trades.records_readable

    if len(trades_df) > 0:
        trade_records = pd.DataFrame({
            "pnl": trades_df.get("PnL", pd.Series(dtype=float)),
            "duration_bars": trades_df.get("Duration", pd.Series(dtype=float)),
        })
    else:
        trade_records = None

    # Buy-and-hold benchmark
    bh_equity = init_cash * (close / close.iloc[0])

    metrics = compute_metrics(
        equity_curve      = equity,
        trades            = trade_records,
        benchmark_equity  = bh_equity,
        bars_per_year     = 252 * 24,
    )

    # Print report
    label = f"{strategy_name} [{period_label}]"
    print(format_report(metrics, label))

    # ── Save results ──────────────────────────────────────────────────────────
    result_path = BACKTESTS_DIR / f"{strategy_name}_{period_label}_results.json"
    save_data = {
        "strategy": strategy_name,
        "period":   period_label,
        "start":    str(df.index[0]),
        "end":      str(df.index[-1]),
        "n_bars":   len(df),
        "metrics":  metrics,
        "pass_criteria": passes_criteria(metrics),
        "verdict":  verdict(metrics),
        "costs": {
            "spread_pips":    SPREAD_PIPS,
            "slippage_pips":  SLIPPAGE_PIPS,
            "commission_rt":  COMMISSION_RT,
        }
    }
    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"[engine] Results saved to {result_path}")

    return metrics, pf


def run_walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    n_windows: int = 4,
) -> Dict[str, Any]:
    """
    Walk-forward validation: split data into n_windows, backtest each.
    Reports stability of performance across windows.
    """
    results = []
    window_size = len(df) // n_windows

    for i in range(n_windows):
        start = i * window_size
        end   = (i + 1) * window_size
        chunk = df.iloc[start:end]

        if chunk["signal_long"].sum() + chunk["signal_short"].sum() < 5:
            continue

        metrics, _ = run_backtest(chunk, strategy_name, period_label=f"wf_{i+1}")
        results.append(metrics)

    if not results:
        return {"walk_forward": "No valid windows"}

    returns = [r.get("annualized_return", 0) for r in results]
    sharpes = [r.get("sharpe_ratio", 0) for r in results]

    summary = {
        "n_windows":      len(results),
        "mean_return":    round(float(np.mean(returns)), 4),
        "std_return":     round(float(np.std(returns)), 4),
        "min_return":     round(float(np.min(returns)), 4),
        "max_return":     round(float(np.max(returns)), 4),
        "mean_sharpe":    round(float(np.mean(sharpes)), 4),
        "pct_positive":   round(float(np.mean([r > 0 for r in returns])), 4),
        "windows":        results,
    }
    return summary
