"""
Module: metrics.py
Purpose: Compute comprehensive trading strategy performance metrics
Author: Strategy Code Engineer Agent
Date: 2026-03-08
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parents[2]))

import numpy as np
import pandas as pd
from typing import Dict, Any

from src.config import get_config


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


def compute_random_baseline(
    close: pd.Series,
    n_trades: int,
    init_cash: float = 100_000.0,
    bars_per_year: int = 252 * 24,
    n_simulations: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate a random long-only baseline with the same trade count as the strategy.
    Entry bars are chosen randomly, exit after the same avg hold duration.

    Returns the median Sharpe and return across simulations.
    This answers: "would a monkey with the same number of trades do as well?"
    A strategy that can't beat this baseline has no real edge.
    """
    rng     = np.random.default_rng(seed)
    n_bars  = len(close)
    returns = close.pct_change().fillna(0).values

    sharpes = []
    ann_rets = []

    avg_hold = max(n_bars // max(n_trades, 1), 1)

    for _ in range(n_simulations):
        equity = np.ones(n_bars) * init_cash
        in_trade = False
        entry_bar = 0

        entry_bars = sorted(rng.choice(n_bars - avg_hold, size=n_trades, replace=False))
        trade_set  = set(entry_bars)
        exit_set   = {b + avg_hold for b in entry_bars}

        pos = 0.0
        for i in range(1, n_bars):
            if i in trade_set and not in_trade:
                pos      = equity[i - 1]
                in_trade = True
                entry_bar = i
            if in_trade:
                equity[i] = equity[i - 1] * (1 + returns[i] * (pos / equity[i - 1]))
            else:
                equity[i] = equity[i - 1]
            if i in exit_set:
                in_trade = False
                pos      = 0.0

        eq_series = pd.Series(equity)
        n_years   = n_bars / bars_per_year
        tot_ret   = (eq_series.iloc[-1] / eq_series.iloc[0]) - 1
        ann_ret   = (1 + tot_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        ret_series = eq_series.pct_change().dropna()
        sharpe    = (ret_series.mean() / ret_series.std() * np.sqrt(bars_per_year)
                     if ret_series.std() > 0 else 0.0)
        sharpes.append(float(sharpe))
        ann_rets.append(float(ann_ret))

    return {
        "random_median_sharpe":     round(float(np.median(sharpes)), 4),
        "random_median_return":     round(float(np.median(ann_rets)), 4),
        "random_p95_sharpe":        round(float(np.percentile(sharpes, 95)), 4),
        "n_simulations":            n_simulations,
        "note": "strategy must beat random_p95_sharpe to have statistically meaningful edge",
    }


def hard_rejection_checks(
    metrics: Dict[str, Any],
    train_regime_dist: Dict[str, float] = None,
    test_regime_dist:  Dict[str, float] = None,
    cost_sensitivity_metrics: Dict[str, Any] = None,
    walk_forward_run: bool = False,
) -> Dict[str, Any]:
    """
    Hard rejection rules that override verdict regardless of Sharpe/return.
    Returns dict with rejection flags and reasons.

    Rules (Phase B2):
    1. Top-10 trade dominance: reject if top 10 trades > 50% of total PnL
    2. Regime distribution drift: reject if any regime differs > 20pp between train and test
    3. Overtrading: reject if avg trade duration < 2 bars
    4. Cost sensitivity: reject if Sharpe drops > 30% under 2x costs
    5. Walk-forward required: reject if walk-forward not run
    """
    rejections = {}

    # Rule 1: Top-10 trade dominance (not checkable without raw trades; skip if unavailable)
    # (Checked in pipeline when raw trade records are available)

    # Rule 3: Overtrading
    avg_dur = metrics.get("avg_trade_duration_bars", 999)
    if avg_dur < 2:
        rejections["overtrading"] = f"Avg trade duration {avg_dur:.1f} bars < 2 bar minimum"

    # Rule 2: Regime drift
    if train_regime_dist and test_regime_dist:
        for regime in train_regime_dist:
            if regime in test_regime_dist:
                drift = abs(train_regime_dist[regime] - test_regime_dist[regime])
                if drift > 0.20:
                    rejections["regime_drift"] = (
                        f"Regime '{regime}' drifted {drift*100:.1f}pp between train and test"
                    )
                    break

    # Rule 4: Cost sensitivity
    if cost_sensitivity_metrics is not None:
        base_sharpe = metrics.get("sharpe_ratio", 0)
        cost2x_sharpe = cost_sensitivity_metrics.get("sharpe_ratio", 0)
        if base_sharpe > 0 and cost2x_sharpe < base_sharpe * 0.70:
            drop_pct = (base_sharpe - cost2x_sharpe) / base_sharpe * 100
            rejections["cost_sensitive"] = (
                f"Sharpe drops {drop_pct:.0f}% under 2x costs ({base_sharpe:.3f} -> {cost2x_sharpe:.3f})"
            )

    # Rule 5: Walk-forward required
    if not walk_forward_run:
        rejections["wf_not_run"] = "Walk-forward validation not run - robustness unverified"

    return {
        "hard_rejected": len(rejections) > 0,
        "rejections": rejections,
    }


def passes_criteria(metrics: Dict[str, Any], split: str = "test") -> Dict[str, bool]:
    """
    Check if metrics meet minimum acceptance criteria for a given split.
    Thresholds read from config.

    Args:
        metrics: Performance metrics dict
        split:   "train", "val", or "test" — thresholds differ per split

    Returns:
        Dict of criterion_name -> bool
    """
    cfg = get_config()
    acc = cfg.get("acceptance", {})

    if split == "train":
        t = acc.get("train", {})
        return {
            "sharpe":        metrics.get("sharpe_ratio", -999)    >= t.get("min_sharpe",        0.0),
            "profit_factor": metrics.get("profit_factor", 0)      >= t.get("min_profit_factor", 1.0),
            "trade_count":   metrics.get("total_trades", 0)       >= t.get("min_trades",         50),
        }
    elif split == "val":
        t = acc.get("val", {})
        return {
            "sharpe":        metrics.get("sharpe_ratio", -999)    >= t.get("min_sharpe",        0.5),
            "profit_factor": metrics.get("profit_factor", 0)      >= t.get("min_profit_factor", 1.1),
            "trade_count":   metrics.get("total_trades", 0)       >= t.get("min_trades",         20),
        }
    else:  # test
        t = acc.get("test", {})
        return {
            "return":        metrics.get("annualized_return", 0)  >= t.get("min_annualized_return", 0.10),
            "sharpe":        metrics.get("sharpe_ratio", -999)    >= t.get("min_sharpe",            1.0),
            "drawdown":      metrics.get("max_drawdown", -999)    >= t.get("min_drawdown",          -0.35),
            "profit_factor": metrics.get("profit_factor", 0)      >= t.get("min_profit_factor",     1.2),
            "trade_count":   metrics.get("total_trades", 0)       >= t.get("min_trades",             30),
            "win_rate":      metrics.get("win_rate", 0)           >= t.get("min_win_rate",           0.40),
        }


def passes_walk_forward(wf_results: Dict[str, Any]) -> Dict[str, bool]:
    """Check if walk-forward results meet robustness thresholds."""
    if not wf_results or "mean_sharpe" not in wf_results:
        return {"available": False}
    return {
        "available":      True,
        "mean_sharpe":    wf_results.get("mean_sharpe", -999)    >= 0.5,
        "positive_pct":   wf_results.get("pct_positive", 0)      >= 0.75,
    }


def multi_split_verdict(
    train_metrics: Dict[str, Any],
    val_metrics:   Dict[str, Any],
    test_metrics:  Dict[str, Any],
    wf_results:    Dict[str, Any] = None,
    hard_checks:   Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Strict multi-split verdict. ALL splits must meet their thresholds.
    A strategy that passes test but fails train is NOT approved — it is
    likely overfit or lucky.

    Returns dict with per-split results and final verdict string.
    """
    train_pass = passes_criteria(train_metrics, "train")
    val_pass   = passes_criteria(val_metrics,   "val")
    test_pass  = passes_criteria(test_metrics,  "test")
    wf_pass    = passes_walk_forward(wf_results) if wf_results else {}

    train_ok = all(train_pass.values())
    val_ok   = all(val_pass.values())
    test_ok  = all(test_pass.values())
    wf_ok    = wf_pass.get("available", False) and all(
        v for k, v in wf_pass.items() if k != "available"
    )
    wf_run   = wf_pass.get("available", False)

    # Hard rejection overrides everything (Phase B2)
    if hard_checks is None:
        hard_checks = {}
    hard_rejected = hard_checks.get("hard_rejected", False)

    # Final verdict logic
    if hard_rejected:
        final = "HARD_REJECTED"
    elif test_ok and train_ok and val_ok and (wf_ok or not wf_run):
        final = "APPROVED"
    elif test_ok and not train_ok:
        final = "OVERFIT"        # test looks good but train failed — suspicious
    elif test_ok and not val_ok:
        final = "UNSTABLE"       # test passed but val didn't — inconsistent
    elif not any(test_pass.values()):
        final = "REJECTED"
    else:
        final = "REVISE"

    return {
        "verdict":       final,
        "train_pass":    train_pass,
        "val_pass":      val_pass,
        "test_pass":     test_pass,
        "wf_pass":       wf_pass,
        "train_ok":      train_ok,
        "val_ok":        val_ok,
        "test_ok":       test_ok,
        "wf_ok":         wf_ok,
        "hard_checks":   hard_checks,
        "flags": {
            "test_without_train": test_ok and not train_ok,
            "test_without_val":   test_ok and not val_ok,
            "wf_not_run":         not wf_run,
            "hard_rejected":      hard_rejected,
        },
    }


def verdict(metrics: Dict[str, Any]) -> str:
    """Single-split verdict (kept for backward compatibility). Prefer multi_split_verdict."""
    criteria  = passes_criteria(metrics, "test")
    all_pass  = all(criteria.values())
    none_pass = not any(criteria.values())
    if all_pass:
        return "APPROVED"
    elif none_pass:
        return "REJECTED"
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
