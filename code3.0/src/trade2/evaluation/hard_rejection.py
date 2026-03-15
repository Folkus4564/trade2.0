"""
evaluation/hard_rejection.py - Hard rejection checks that override verdict.
"""

from typing import Any, Dict, Optional


def hard_rejection_checks(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    train_regime_dist: Dict[str, float] = None,
    test_regime_dist:  Dict[str, float] = None,
    cost_sensitivity_metrics: Dict[str, Any] = None,
    walk_forward_run: bool = False,
    train_metrics: Dict[str, Any] = None,
    val_metrics: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Hard rejection rules that override verdict regardless of Sharpe/return.

    Rules:
    1. Overtrading: avg trade duration < min_avg_trade_duration_bars
    2. Regime distribution drift: any regime > max_regime_drift_pp difference between train/test
    3. Cost sensitivity: Sharpe drops > max_cost_sharpe_drop_pct under 2x costs
    4. Walk-forward not run (when require_walk_forward is true)

    All thresholds read from config["hard_rejection"].
    """
    hr_cfg = config["hard_rejection"]
    min_dur             = hr_cfg["min_avg_trade_duration_bars"]
    max_drift           = hr_cfg["max_regime_drift_pp"]
    max_drop_pct        = hr_cfg["max_cost_sharpe_drop_pct"]
    require_wf          = hr_cfg["require_walk_forward"]
    min_trades_train    = hr_cfg["min_trades_train"]
    min_trades_val      = hr_cfg["min_trades_val"]
    max_overfit_ratio   = hr_cfg["max_overfitting_sharpe_ratio"]

    rejections = {}

    # Rule 1: Overtrading
    avg_dur = metrics.get("avg_trade_duration_bars", 999)
    if avg_dur < min_dur:
        rejections["overtrading"] = (
            f"Avg trade duration {avg_dur:.1f} bars < {min_dur} bar minimum"
        )

    # Rule 2: Regime drift
    if train_regime_dist and test_regime_dist:
        for regime in train_regime_dist:
            if regime in test_regime_dist:
                drift = abs(train_regime_dist[regime] - test_regime_dist[regime])
                if drift > max_drift:
                    rejections["regime_drift"] = (
                        f"Regime '{regime}' drifted {drift*100:.1f}pp between train and test"
                    )
                    break

    # Rule 3: Cost sensitivity — reject if Sharpe drops more than max_drop_pct under 2x costs
    if cost_sensitivity_metrics is not None:
        base_sharpe   = metrics.get("sharpe_ratio", 0)
        cost2x_sharpe = cost_sensitivity_metrics.get("sharpe_ratio", 0)
        if base_sharpe > 0 and cost2x_sharpe < base_sharpe * (1.0 - max_drop_pct):
            drop_pct = (base_sharpe - cost2x_sharpe) / base_sharpe * 100
            rejections["cost_sensitive"] = (
                f"Sharpe drops {drop_pct:.0f}% under 2x costs ({base_sharpe:.3f} -> {cost2x_sharpe:.3f})"
            )

    # Rule 4: Walk-forward required
    if require_wf and not walk_forward_run:
        rejections["wf_not_run"] = "Walk-forward validation not run - robustness unverified"

    # Rule 5: Minimum trades on train split (Phase 6 - cross-split validation)
    if train_metrics is not None:
        train_trades = train_metrics.get("total_trades", 0)
        if train_trades < min_trades_train:
            rejections["train_too_few_trades"] = (
                f"Train split only {train_trades} trades < {min_trades_train} minimum"
            )

    # Rule 6: Minimum trades on val split
    if val_metrics is not None:
        val_trades = val_metrics.get("total_trades", 0)
        if val_trades < min_trades_val:
            rejections["val_too_few_trades"] = (
                f"Val split only {val_trades} trades < {min_trades_val} minimum"
            )

    # Rule 7: Overfitting check — test Sharpe must not be >> train Sharpe
    if train_metrics is not None:
        test_sharpe  = metrics.get("sharpe_ratio", 0)
        train_sharpe = train_metrics.get("sharpe_ratio", 0)
        if train_sharpe > 0 and test_sharpe > train_sharpe * max_overfit_ratio:
            rejections["sharpe_overfit"] = (
                f"Test Sharpe ({test_sharpe:.3f}) is {test_sharpe/train_sharpe:.1f}x train Sharpe "
                f"({train_sharpe:.3f}) — likely overfit (max ratio {max_overfit_ratio})"
            )

    return {
        "hard_rejected": len(rejections) > 0,
        "rejections":    rejections,
    }
