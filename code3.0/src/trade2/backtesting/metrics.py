"""
backtesting/metrics.py - Performance metrics computation (pure computation, no evaluation).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame = None,
    benchmark_equity: pd.Series = None,
    risk_free_rate: float = 0.04,
    bars_per_year: int = 252 * 24,
) -> Dict[str, Any]:
    """
    Compute full performance metrics from an equity curve.

    Args:
        equity_curve:     Portfolio value series
        trades:           DataFrame with 'pnl' and 'duration_bars' columns (optional)
        benchmark_equity: Buy-and-hold equity curve for comparison (optional)
        risk_free_rate:   Annual risk-free rate
        bars_per_year:    252*24 for 1H, 252*24*12 for 5M

    Returns:
        Dict of all performance metrics.
    """
    returns = equity_curve.pct_change().dropna()

    total_return      = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years           = len(returns) / bars_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    ann_vol    = returns.std() * np.sqrt(bars_per_year)
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    max_dd      = drawdown.min()
    calmar      = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

    excess_ret = returns - (risk_free_rate / bars_per_year)
    sharpe     = (excess_ret.mean() / returns.std() * np.sqrt(bars_per_year)
                  if returns.std() > 0 else 0.0)

    neg_ret  = returns[returns < 0]
    down_std = neg_ret.std() * np.sqrt(bars_per_year) if len(neg_ret) > 0 else 1e-10
    sortino  = (annualized_return - risk_free_rate) / down_std if down_std > 0 else 0.0

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

    if trades is not None and len(trades) > 0:
        if "pnl" in trades.columns:
            pnl      = trades["pnl"]
            wins     = pnl[pnl > 0]
            losses   = pnl[pnl < 0]
            win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
            avg_win  = wins.mean()   if len(wins)   > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0
            pf       = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
            wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
            metrics.update({
                "total_trades":       int(len(pnl)),
                "win_rate":           round(float(win_rate), 4),
                "profit_factor":      round(float(pf), 4),
                "avg_win_loss_ratio": round(float(wl_ratio), 4),
                "avg_trade_pnl":      round(float(pnl.mean()), 4),
            })
        if "duration_bars" in trades.columns:
            metrics["avg_trade_duration_bars"] = round(float(trades["duration_bars"].mean()), 2)

    if benchmark_equity is not None:
        bench_ret   = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1
        bench_n_yrs = len(benchmark_equity.pct_change().dropna()) / bars_per_year
        bench_ann   = (1 + bench_ret) ** (1 / bench_n_yrs) - 1 if bench_n_yrs > 0 else 0.0
        alpha       = annualized_return - bench_ann
        bench_vol   = benchmark_equity.pct_change().dropna().std() * np.sqrt(bars_per_year)
        ir          = alpha / bench_vol if bench_vol > 0 else 0.0
        metrics.update({
            "benchmark_return":   round(float(bench_ann), 4),
            "alpha_vs_benchmark": round(float(alpha), 4),
            "information_ratio":  round(float(ir), 4),
        })

    return metrics


def compute_random_baseline(
    close: pd.Series,
    n_trades: int,
    init_cash: float = 100_000.0,
    bars_per_year: int = 252 * 24,
    n_simulations: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate random long-only baseline with same trade count as strategy.
    A strategy that can't beat p95 of this has no real edge.
    """
    rng     = np.random.default_rng(seed)
    n_bars  = len(close)
    returns = close.pct_change().fillna(0).values

    sharpes, ann_rets = [], []
    avg_hold = max(n_bars // max(n_trades, 1), 1)

    population = max(n_bars - avg_hold, n_trades)  # guard: never sample more than population
    for _ in range(n_simulations):
        equity = np.ones(n_bars) * init_cash
        entry_bars = sorted(rng.choice(population, size=n_trades, replace=False))
        trade_set  = set(entry_bars)
        exit_set   = {b + avg_hold for b in entry_bars}

        in_trade, pos = False, 0.0
        for i in range(1, n_bars):
            if i in trade_set and not in_trade:
                pos = equity[i - 1]
                in_trade = True
            equity[i] = equity[i-1] * (1 + returns[i] * (pos / equity[i-1])) if in_trade else equity[i-1]
            if i in exit_set:
                in_trade, pos = False, 0.0

        eq = pd.Series(equity)
        n_yrs   = n_bars / bars_per_year
        tot_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
        ann_ret = (1 + tot_ret) ** (1 / n_yrs) - 1 if n_yrs > 0 else 0.0
        ret_s   = eq.pct_change().dropna()
        sharpe  = (ret_s.mean() / ret_s.std() * np.sqrt(bars_per_year) if ret_s.std() > 0 else 0.0)
        sharpes.append(float(sharpe))
        ann_rets.append(float(ann_ret))

    return {
        "random_median_sharpe": round(float(np.median(sharpes)), 4),
        "random_median_return": round(float(np.median(ann_rets)), 4),
        "random_p95_sharpe":    round(float(np.percentile(sharpes, 95)), 4),
        "n_simulations":        n_simulations,
    }


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
        f"  Ann. Volatility   : {metrics.get('annualized_volatility', 0)*100:>8.2f}%",
        f"{DASH}",
        f"  Total Trades      : {metrics.get('total_trades', 'N/A'):>8}",
        f"  Win Rate          : {metrics.get('win_rate', 0)*100:>8.2f}%",
        f"  Profit Factor     : {metrics.get('profit_factor', 0):>8.3f}",
        f"  Avg W/L Ratio     : {metrics.get('avg_win_loss_ratio', 0):>8.3f}",
    ]
    if "benchmark_return" in metrics:
        lines += [
            f"{DASH}",
            f"  Benchmark Return  : {metrics.get('benchmark_return', 0)*100:>8.2f}%",
            f"  Alpha             : {metrics.get('alpha_vs_benchmark', 0)*100:>8.2f}%",
            f"  Information Ratio : {metrics.get('information_ratio', 0):>8.3f}",
        ]
    lines += [f"{SEP}\n"]
    return "\n".join(lines)
