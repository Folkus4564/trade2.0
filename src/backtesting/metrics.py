"""
Module: metrics.py
Purpose: Compute comprehensive trading strategy performance metrics
Author: Strategy Code Engineer Agent
Date: 2026-03-08
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame = None,
    benchmark_equity: pd.Series = None,
    risk_free_rate: float = 0.04,
    bars_per_year: int = 252 * 24,  # hourly bars
) -> Dict[str, Any]:
    """
    Compute full performance metrics from an equity curve.

    Args:
        equity_curve:     Portfolio value series (DatetimeIndex)
        trades:           DataFrame with trade-level detail (optional)
        benchmark_equity: Buy-and-hold equity curve for comparison (optional)
        risk_free_rate:   Annual risk-free rate for Sharpe/Sortino
        bars_per_year:    Number of bars in a year (252*24 for 1H)

    Returns:
        Dictionary of all performance metrics
    """
    returns = equity_curve.pct_change().dropna()

    # ── Return metrics ────────────────────────────────────────────────────────
    total_return      = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years           = len(returns) / bars_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # ── Risk metrics ──────────────────────────────────────────────────────────
    ann_vol = returns.std() * np.sqrt(bars_per_year)

    # Drawdown
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # Calmar
    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

    # Sharpe
    excess_ret = returns - (risk_free_rate / bars_per_year)
    sharpe     = (excess_ret.mean() / returns.std() * np.sqrt(bars_per_year)
                  if returns.std() > 0 else 0.0)

    # Sortino (downside deviation only)
    neg_ret    = returns[returns < 0]
    down_std   = neg_ret.std() * np.sqrt(bars_per_year) if len(neg_ret) > 0 else 1e-10
    sortino    = (annualized_return - risk_free_rate) / down_std if down_std > 0 else 0.0

    metrics = {
        "annualized_return":    round(float(annualized_return), 4),
        "total_return":         round(float(total_return), 4),
        "annualized_volatility": round(float(ann_vol), 4),
        "sharpe_ratio":         round(float(sharpe), 4),
        "sortino_ratio":        round(float(sortino), 4),
        "max_drawdown":         round(float(max_dd), 4),
        "calmar_ratio":         round(float(calmar), 4),
        "n_years":              round(float(n_years), 2),
    }

    # ── Trade-level metrics (if trades provided) ──────────────────────────────
    if trades is not None and len(trades) > 0:
        if "pnl" in trades.columns:
            pnl        = trades["pnl"]
            wins       = pnl[pnl > 0]
            losses     = pnl[pnl < 0]
            win_rate   = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
            avg_win    = wins.mean()   if len(wins)   > 0 else 0.0
            avg_loss   = losses.mean() if len(losses) > 0 else 0.0
            profit_fac = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
            avg_wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
            metrics.update({
                "total_trades":      int(len(pnl)),
                "win_rate":          round(float(win_rate), 4),
                "profit_factor":     round(float(profit_fac), 4),
                "avg_win_loss_ratio": round(float(avg_wl_ratio), 4),
                "avg_trade_pnl":     round(float(pnl.mean()), 4),
            })
        if "duration_bars" in trades.columns:
            metrics["avg_trade_duration_bars"] = round(float(trades["duration_bars"].mean()), 2)

    # ── Benchmark comparison ──────────────────────────────────────────────────
    if benchmark_equity is not None:
        bench_ret   = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1
        bench_n_yrs = len(benchmark_equity.pct_change().dropna()) / bars_per_year
        bench_ann   = (1 + bench_ret) ** (1 / bench_n_yrs) - 1 if bench_n_yrs > 0 else 0.0
        alpha       = annualized_return - bench_ann
        bench_vol   = benchmark_equity.pct_change().dropna().std() * np.sqrt(bars_per_year)
        ir          = alpha / bench_vol if bench_vol > 0 else 0.0
        metrics.update({
            "benchmark_return":  round(float(bench_ann), 4),
            "alpha_vs_benchmark": round(float(alpha), 4),
            "information_ratio": round(float(ir), 4),
        })

    return metrics


def passes_criteria(metrics: Dict[str, Any]) -> Dict[str, bool]:
    """Check if metrics meet minimum acceptance criteria."""
    return {
        "return":       metrics.get("annualized_return", 0)  >= 0.10,
        "sharpe":       metrics.get("sharpe_ratio", 0)       >= 1.0,
        "drawdown":     metrics.get("max_drawdown", -999)    >= -0.35,
        "profit_factor": metrics.get("profit_factor", 0)     >= 1.2,
        "trade_count":  metrics.get("total_trades", 0)       >= 30,
        "win_rate":     metrics.get("win_rate", 0)            >= 0.40,
    }


def verdict(metrics: Dict[str, Any]) -> str:
    """Return APPROVED / REVISE / REJECTED based on metrics."""
    criteria = passes_criteria(metrics)
    all_pass = all(criteria.values())
    none_pass = not any(criteria.values())

    if all_pass:
        return "APPROVED"
    elif none_pass:
        return "REJECTED"
    else:
        return "REVISE"


def format_report(metrics: Dict[str, Any], strategy_name: str = "Strategy") -> str:
    """Return a formatted text performance report."""
    SEP  = "=" * 50
    DASH = "-" * 50
    lines = [
        f"\n{SEP}",
        f"  PERFORMANCE REPORT: {strategy_name}",
        f"{SEP}",
        f"  Annualized Return : {metrics.get('annualized_return', 0)*100:>8.2f}%",
        f"  Sharpe Ratio      : {metrics.get('sharpe_ratio', 0):>8.3f}",
        f"  Sortino Ratio     : {metrics.get('sortino_ratio', 0):>8.3f}",
        f"  Max Drawdown      : {metrics.get('max_drawdown', 0)*100:>8.2f}%",
        f"  Calmar Ratio      : {metrics.get('calmar_ratio', 0):>8.3f}",
        f"  Annualized Vol    : {metrics.get('annualized_volatility', 0)*100:>8.2f}%",
        f"{DASH}",
        f"  Total Trades      : {metrics.get('total_trades', 'N/A'):>8}",
        f"  Win Rate          : {metrics.get('win_rate', 0)*100:>8.2f}%",
        f"  Profit Factor     : {metrics.get('profit_factor', 0):>8.3f}",
        f"  Avg Win/Loss Ratio: {metrics.get('avg_win_loss_ratio', 0):>8.3f}",
    ]
    if "benchmark_return" in metrics:
        lines += [
            f"{DASH}",
            f"  Benchmark Return  : {metrics.get('benchmark_return', 0)*100:>8.2f}%",
            f"  Alpha             : {metrics.get('alpha_vs_benchmark', 0)*100:>8.2f}%",
            f"  Information Ratio : {metrics.get('information_ratio', 0):>8.3f}",
        ]
    criteria = passes_criteria(metrics)
    v = verdict(metrics)
    lines += [
        f"{SEP}",
        f"  Verdict: {v}",
        f"  Criteria passed: {sum(criteria.values())}/{len(criteria)}",
        f"{SEP}\n",
    ]
    return "\n".join(lines)
