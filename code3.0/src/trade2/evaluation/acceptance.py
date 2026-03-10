"""
evaluation/acceptance.py - Check metrics against acceptance thresholds from config.
"""

from typing import Any, Dict


def passes_criteria(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    split: str = "test",
) -> Dict[str, bool]:
    """
    Check if metrics meet minimum acceptance criteria for a given split.

    Args:
        metrics: Performance metrics dict
        config:  Full config dict (thresholds read from acceptance section)
        split:   'train', 'val', or 'test'

    Returns:
        Dict of criterion_name -> bool
    """
    acc = config.get("acceptance", {})

    if split == "train":
        t = acc.get("train", {})
        return {
            "sharpe":        metrics.get("sharpe_ratio", -999)  >= t.get("min_sharpe",        0.0),
            "profit_factor": metrics.get("profit_factor", 0)    >= t.get("min_profit_factor", 1.0),
            "trade_count":   metrics.get("total_trades", 0)     >= t.get("min_trades",         50),
        }
    elif split == "val":
        t = acc.get("val", {})
        return {
            "sharpe":        metrics.get("sharpe_ratio", -999)  >= t.get("min_sharpe",        0.5),
            "profit_factor": metrics.get("profit_factor", 0)    >= t.get("min_profit_factor", 1.1),
            "trade_count":   metrics.get("total_trades", 0)     >= t.get("min_trades",         20),
        }
    else:  # test
        t = acc.get("test", {})
        return {
            "return":        metrics.get("annualized_return", 0) >= t.get("min_annualized_return", 0.10),
            "sharpe":        metrics.get("sharpe_ratio", -999)   >= t.get("min_sharpe",            1.0),
            "drawdown":      metrics.get("max_drawdown", -999)   >= t.get("min_drawdown",          -0.35),
            "profit_factor": metrics.get("profit_factor", 0)     >= t.get("min_profit_factor",     1.2),
            "trade_count":   metrics.get("total_trades", 0)      >= t.get("min_trades",             30),
            "win_rate":      metrics.get("win_rate", 0)          >= t.get("min_win_rate",           0.40),
        }


def passes_walk_forward(
    wf_results: Dict[str, Any],
    config: Dict[str, Any] = None,
) -> Dict[str, bool]:
    """Check if walk-forward results meet robustness thresholds."""
    if not wf_results or "mean_sharpe" not in wf_results:
        return {"available": False}
    acc = (config or {}).get("acceptance", {}).get("walk_forward", {})
    return {
        "available":    True,
        "mean_sharpe":  wf_results.get("mean_sharpe", -999) >= acc.get("min_mean_sharpe",       0.5),
        "positive_pct": wf_results.get("pct_positive", 0)   >= acc.get("min_positive_windows", 0.75),
    }
