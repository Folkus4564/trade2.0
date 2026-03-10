"""
Module: engine.py
Purpose: vectorbt backtest engine for 5M SMC strategy
         Spread cost fix applied (spread + slippage both charged)
"""

import json
import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path
from typing import Dict, Any, Tuple

ROOT          = Path(__file__).parents[2]
BACKTESTS_DIR = ROOT / "backtests_v2"
BACKTESTS_DIR.mkdir(exist_ok=True)


# Re-import metrics from v1 (no need to duplicate)
import sys
sys.path.insert(0, str(ROOT))
from src.backtesting.metrics import compute_metrics, format_report, verdict, passes_criteria


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    period_label: str = "test",
    init_cash: float = 100_000.0,
    size_pct: float = 0.95,
    config: dict = None,
) -> Tuple[Dict[str, Any], Any]:
    """
    Run vectorbt backtest on 5M signal DataFrame.

    Expects columns: signal_long, signal_short, exit_long, exit_short
    """
    if not config or "costs" not in config:
        raise ValueError(
            "[engine_v2] config with a 'costs' section is required. "
            "Pass the loaded config dict to run_backtest()."
        )
    try:
        c = config["costs"]
        spread_pips   = c["spread_pips"]
        slippage_pips = c["slippage_pips"]
        commission_rt = c["commission_rt"]
    except KeyError as e:
        raise ValueError(
            f"[engine_v2] Missing config key: {e}. "
            "Ensure config.yaml has costs.spread_pips, costs.slippage_pips, costs.commission_rt."
        ) from e

    close = df["Close"].astype(float)

    long_entries  = df["signal_long"].astype(bool)
    long_exits    = df["exit_long"].astype(bool)
    short_entries = df["signal_short"].astype(bool)
    short_exits   = df["exit_short"].astype(bool)

    conflict      = long_entries & short_entries
    long_entries  = long_entries  & ~conflict
    short_entries = short_entries & ~conflict

    if long_entries.sum() == 0 and short_entries.sum() == 0:
        print(f"[engine_v2] WARNING: No signals for {period_label}")
        return {
            "annualized_return": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "total_trades": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
        }, None

    # Spread + slippage both applied (bug fix vs v1)
    avg_price  = close.mean()
    pip_value  = 0.01
    slippage   = ((spread_pips + slippage_pips) * pip_value) / avg_price

    trade_value = init_cash * size_pct

    pf = vbt.Portfolio.from_signals(
        close         = close,
        entries       = long_entries,
        exits         = long_exits,
        short_entries = short_entries,
        short_exits   = short_exits,
        init_cash     = init_cash,
        fees          = commission_rt / 2,
        slippage      = slippage,
        size          = trade_value,
        size_type     = "value",
        freq          = "5min",
    )

    equity    = pf.value()
    trades_df = pf.trades.records_readable

    trade_records = None
    if len(trades_df) > 0:
        trade_records = pd.DataFrame({
            "pnl":           trades_df.get("PnL", pd.Series(dtype=float)),
            "duration_bars": trades_df.get("Duration", pd.Series(dtype=float)),
        })

    bh_equity = init_cash * (close / close.iloc[0])

    # 5M bars per year: 252 trading days * 24h * 12 bars/hour
    bars_per_year = 252 * 24 * 12

    metrics = compute_metrics(
        equity_curve     = equity,
        trades           = trade_records,
        benchmark_equity = bh_equity,
        bars_per_year    = bars_per_year,
    )

    print(format_report(metrics, f"{strategy_name} [{period_label}]"))

    result_path = BACKTESTS_DIR / f"{strategy_name}_{period_label}_results.json"
    with open(result_path, "w") as f:
        json.dump({
            "strategy": strategy_name,
            "period":   period_label,
            "start":    str(df.index[0]),
            "end":      str(df.index[-1]),
            "n_bars":   len(df),
            "metrics":  metrics,
            "pass_criteria": passes_criteria(metrics),
            "verdict":  verdict(metrics),
            "costs": {
                "spread_pips":   spread_pips,
                "slippage_pips": slippage_pips,
                "commission_rt": commission_rt,
            },
        }, f, indent=2)
    print(f"[engine_v2] Results saved to {result_path}")

    return metrics, pf
