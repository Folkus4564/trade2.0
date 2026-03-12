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
        config:  Full config dict (thresholds read from config["acceptance"])
        split:   'train', 'val', or 'test'

    Returns:
        Dict of criterion_name -> bool
    """
    acc = config["acceptance"]

    if split == "train":
        t = acc["train"]
        return {
            "sharpe":        metrics.get("sharpe_ratio", -999)  >= t["min_sharpe"],
            "profit_factor": metrics.get("profit_factor", 0)    >= t["min_profit_factor"],
            "trade_count":   metrics.get("total_trades", 0)     >= t["min_trades"],
        }
    elif split == "val":
        t = acc["val"]
        return {
            "sharpe":        metrics.get("sharpe_ratio", -999)  >= t["min_sharpe"],
            "profit_factor": metrics.get("profit_factor", 0)    >= t["min_profit_factor"],
            "trade_count":   metrics.get("total_trades", 0)     >= t["min_trades"],
        }
    else:  # test
        t = acc["test"]
        return {
            "return":        metrics.get("annualized_return", 0) >= t["min_annualized_return"],
            "sharpe":        metrics.get("sharpe_ratio", -999)   >= t["min_sharpe"],
            "drawdown":      metrics.get("max_drawdown", -999)   >= t["min_drawdown"],
            "profit_factor": metrics.get("profit_factor", 0)     >= t["min_profit_factor"],
            "trade_count":   metrics.get("total_trades", 0)      >= t["min_trades"],
            "win_rate":      metrics.get("win_rate", 0)          >= t["min_win_rate"],
        }


def passes_walk_forward(
    wf_results: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, bool]:
    """Check if walk-forward results meet robustness thresholds."""
    if not wf_results or "mean_sharpe" not in wf_results:
        return {"available": False}
    acc = config["acceptance"]["walk_forward"]
    return {
        "available":    True,
        "mean_sharpe":  wf_results.get("mean_sharpe", -999) >= acc["min_mean_sharpe"],
        "positive_pct": wf_results.get("pct_positive", 0)   >= acc["min_positive_windows"],
    }
