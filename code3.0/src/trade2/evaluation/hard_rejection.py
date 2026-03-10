"""
evaluation/hard_rejection.py - Hard rejection checks that override verdict.
"""

from typing import Any, Dict, Optional


def hard_rejection_checks(
    metrics: Dict[str, Any],
    train_regime_dist: Dict[str, float] = None,
    test_regime_dist:  Dict[str, float] = None,
    cost_sensitivity_metrics: Dict[str, Any] = None,
    walk_forward_run: bool = False,
) -> Dict[str, Any]:
    """
    Hard rejection rules that override verdict regardless of Sharpe/return.

    Rules:
    1. Overtrading: avg trade duration < 2 bars
    2. Regime distribution drift: any regime > 20pp difference between train/test
    3. Cost sensitivity: Sharpe drops > 30% under 2x costs
    4. Walk-forward not run
    """
    rejections = {}

    # Rule 1: Overtrading
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

    # Rule 3: Cost sensitivity
    if cost_sensitivity_metrics is not None:
        base_sharpe  = metrics.get("sharpe_ratio", 0)
        cost2x_sharpe = cost_sensitivity_metrics.get("sharpe_ratio", 0)
        if base_sharpe > 0 and cost2x_sharpe < base_sharpe * 0.70:
            drop_pct = (base_sharpe - cost2x_sharpe) / base_sharpe * 100
            rejections["cost_sensitive"] = (
                f"Sharpe drops {drop_pct:.0f}% under 2x costs ({base_sharpe:.3f} -> {cost2x_sharpe:.3f})"
            )

    # Rule 4: Walk-forward required
    if not walk_forward_run:
        rejections["wf_not_run"] = "Walk-forward validation not run - robustness unverified"

    return {
        "hard_rejected": len(rejections) > 0,
        "rejections":    rejections,
    }
