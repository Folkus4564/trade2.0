"""
live/reporter.py - Performance report from live trade log CSV.

Reads the CSV written by TradeLogger, computes standard metrics
(annualized return, Sharpe, max drawdown, win rate, profit factor),
and compares against backtest metrics. Writes JSON + text reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Backtest reference metrics for comparison (from approved strategy metrics.json)
_BACKTEST_REF: Dict[str, Dict[str, float]] = {
    "hmm1h_smc5m_89pct": {
        "annualized_return": 0.8933,
        "sharpe_ratio":      4.10,
        "max_drawdown":     -0.0753,
        "win_rate":          0.5781,
        "profit_factor":     3.2904,
        "total_trades":      128,
    },
    "hmm1h_smc5m_tp2x_49pct": {
        "annualized_return": 0.49,
        "sharpe_ratio":      3.16,
        "max_drawdown":     -0.10,
        "win_rate":          0.50,
        "profit_factor":     2.0,
        "total_trades":      50,
    },
}


def generate_report(
    trade_log_path: Path,
    report_dir: Path,
    strategy_name: str,
    initial_equity: float = 10_000.0,
    backtest_ref: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute live performance metrics and write JSON + text reports.

    Args:
        trade_log_path: Path to the trade log CSV
        report_dir:     Directory to write reports
        strategy_name:  Strategy identifier (for file naming)
        initial_equity: Starting equity for return calculation
        backtest_ref:   Reference backtest metrics for comparison

    Returns:
        Metrics dict
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load trade log
    df = pd.read_csv(trade_log_path) if Path(trade_log_path).exists() else pd.DataFrame()

    if df.empty or "pnl" not in df.columns:
        logger.warning(f"[Reporter] No trades yet for {strategy_name}")
        metrics = _empty_metrics(strategy_name)
        _write_reports(metrics, report_dir, strategy_name)
        return metrics

    df = df.dropna(subset=["pnl"])
    if len(df) == 0:
        metrics = _empty_metrics(strategy_name)
        _write_reports(metrics, report_dir, strategy_name)
        return metrics

    # Equity curve from cumulative P&L
    equity = initial_equity + df["pnl"].cumsum()
    equity_series = pd.Series(equity.values, name="equity")

    # Basic stats
    total_trades  = len(df)
    wins          = (df["pnl"] > 0).sum()
    losses        = (df["pnl"] <= 0).sum()
    win_rate      = wins / total_trades if total_trades > 0 else 0.0
    avg_win       = df.loc[df["pnl"] > 0, "pnl"].mean() if wins > 0 else 0.0
    avg_loss      = abs(df.loc[df["pnl"] <= 0, "pnl"].mean()) if losses > 0 else 1e-9
    profit_factor = (wins * avg_win) / max(losses * avg_loss, 1e-9)

    # Annualized return
    total_return  = (equity_series.iloc[-1] - initial_equity) / initial_equity
    # Estimate annualization from duration_minutes if available
    if "duration_minutes" in df.columns and df["duration_minutes"].notna().any():
        total_minutes = df["duration_minutes"].dropna().sum()
        n_years       = total_minutes / (252 * 24 * 60)
    elif "entry_time" in df.columns and "exit_time" in df.columns:
        try:
            t0 = pd.Timestamp(df["entry_time"].dropna().iloc[0])
            t1 = pd.Timestamp(df["exit_time"].dropna().iloc[-1])
            n_years = (t1 - t0).total_seconds() / (365.25 * 24 * 3600)
        except Exception:
            n_years = 0.1
    else:
        n_years = 0.1

    n_years       = max(n_years, 1 / 252)
    ann_return    = (1 + total_return) ** (1 / n_years) - 1

    # Sharpe (daily P&L)
    daily_pnl = equity_series.diff().dropna()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    roll_max  = equity_series.cummax()
    drawdown  = (equity_series - roll_max) / roll_max
    max_dd    = float(drawdown.min())

    # Avg trade metrics
    avg_pnl      = float(df["pnl"].mean())
    total_pnl    = float(df["pnl"].sum())
    avg_duration = float(df["duration_minutes"].mean()) if "duration_minutes" in df.columns else None

    metrics = {
        "strategy_name":     strategy_name,
        "report_time":       datetime.utcnow().isoformat(),
        "total_trades":      total_trades,
        "total_pnl":         round(total_pnl, 2),
        "total_return":      round(total_return, 4),
        "annualized_return": round(ann_return, 4),
        "sharpe_ratio":      round(sharpe, 4),
        "max_drawdown":      round(max_dd, 4),
        "win_rate":          round(win_rate, 4),
        "profit_factor":     round(profit_factor, 4),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "avg_duration_min":  round(avg_duration, 1) if avg_duration else None,
        "n_years_live":      round(n_years, 3),
    }

    # Add backtest comparison
    ref = backtest_ref or _BACKTEST_REF.get(strategy_name, {})
    if ref:
        divergence = {}
        for k, v in ref.items():
            if k in metrics and v != 0:
                live_val = metrics[k]
                divergence[k] = {
                    "live":      live_val,
                    "backtest":  v,
                    "delta_pct": round((live_val - v) / abs(v) * 100, 1) if v != 0 else None,
                }
        metrics["backtest_comparison"] = divergence

    _write_reports(metrics, report_dir, strategy_name)
    return metrics


def _empty_metrics(strategy_name: str) -> Dict[str, Any]:
    return {
        "strategy_name":     strategy_name,
        "report_time":       datetime.utcnow().isoformat(),
        "total_trades":      0,
        "total_pnl":         0.0,
        "total_return":      0.0,
        "annualized_return": 0.0,
        "sharpe_ratio":      0.0,
        "max_drawdown":      0.0,
        "win_rate":          0.0,
        "profit_factor":     0.0,
        "avg_pnl_per_trade": 0.0,
        "avg_duration_min":  None,
        "n_years_live":      0.0,
        "note":              "No completed trades yet",
    }


def _write_reports(metrics: Dict[str, Any], report_dir: Path, strategy_name: str) -> None:
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"live_report_{strategy_name}_{ts}"

    # JSON
    json_path = report_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Human-readable text
    txt_path = report_dir / f"{stem}.txt"
    lines = [
        f"Live Performance Report — {strategy_name}",
        f"Generated: {metrics['report_time']}",
        "=" * 55,
        f"  Trades:             {metrics['total_trades']}",
        f"  Total P&L:          ${metrics['total_pnl']:,.2f}",
        f"  Total Return:       {metrics['total_return']*100:.2f}%",
        f"  Ann. Return:        {metrics['annualized_return']*100:.2f}%",
        f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}",
        f"  Max Drawdown:       {metrics['max_drawdown']*100:.2f}%",
        f"  Win Rate:           {metrics['win_rate']*100:.1f}%",
        f"  Profit Factor:      {metrics['profit_factor']:.2f}",
        f"  Avg P&L/Trade:      ${metrics['avg_pnl_per_trade']:.2f}",
    ]
    if metrics.get("avg_duration_min"):
        lines.append(f"  Avg Duration:       {metrics['avg_duration_min']}min")

    cmp = metrics.get("backtest_comparison", {})
    if cmp:
        lines.append("")
        lines.append("Backtest Comparison:")
        for k, v in cmp.items():
            delta = v.get("delta_pct")
            flag  = " *** DIVERGENCE ***" if delta is not None and abs(delta) > 30 else ""
            lines.append(
                f"  {k:25s} live={v['live']:>8.3f}  bt={v['backtest']:>8.3f}  "
                f"delta={delta:>+6.1f}%{flag}"
            )

    txt_path.write_text("\n".join(lines) + "\n")
    logger.info(f"[Reporter] Report written to {json_path} and {txt_path}")
