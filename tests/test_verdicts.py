"""
test_verdicts.py - Tests for acceptance criteria, hard rejection, and verdict logic.
"""

import pytest
from trade2.evaluation.acceptance import passes_criteria, passes_walk_forward
from trade2.evaluation.hard_rejection import hard_rejection_checks
from trade2.evaluation.verdict import multi_split_verdict, verdict


# ---- passes_criteria ----

def test_passes_criteria_test_all_pass(base_config):
    metrics = {
        "annualized_return": 0.25, "sharpe_ratio": 1.5, "max_drawdown": -0.15,
        "profit_factor": 1.8, "total_trades": 80, "win_rate": 0.55,
    }
    result = passes_criteria(metrics, base_config, "test")
    assert all(result.values()), f"Expected all pass, got: {result}"


def test_passes_criteria_test_fail_sharpe(base_config):
    metrics = {
        "annualized_return": 0.25, "sharpe_ratio": 0.5, "max_drawdown": -0.15,
        "profit_factor": 1.8, "total_trades": 80, "win_rate": 0.55,
    }
    result = passes_criteria(metrics, base_config, "test")
    assert not result["sharpe"], "Sharpe < 1.0 should fail"


def test_passes_criteria_test_fail_trades(base_config):
    metrics = {
        "annualized_return": 0.25, "sharpe_ratio": 1.5, "max_drawdown": -0.15,
        "profit_factor": 1.8, "total_trades": 10, "win_rate": 0.55,
    }
    result = passes_criteria(metrics, base_config, "test")
    assert not result["trade_count"], "Only 10 trades should fail"


def test_passes_criteria_train_split(base_config):
    metrics = {"sharpe_ratio": 0.5, "profit_factor": 1.2, "total_trades": 100}
    result  = passes_criteria(metrics, base_config, "train")
    assert result["sharpe"]
    assert result["profit_factor"]
    assert result["trade_count"]


# ---- hard_rejection_checks ----

def test_hard_rejection_overtrading():
    metrics = {"avg_trade_duration_bars": 1.5, "sharpe_ratio": 2.0}
    result  = hard_rejection_checks(metrics)
    assert result["hard_rejected"]
    assert "overtrading" in result["rejections"]


def test_hard_rejection_regime_drift():
    metrics = {"avg_trade_duration_bars": 10}
    train_dist = {"bull": 0.40, "bear": 0.30, "sideways": 0.30}
    test_dist  = {"bull": 0.10, "bear": 0.60, "sideways": 0.30}  # bull drifted 30pp
    result = hard_rejection_checks(metrics, train_dist, test_dist)
    assert result["hard_rejected"]
    assert "regime_drift" in result["rejections"]


def test_hard_rejection_cost_sensitivity():
    base_metrics = {"avg_trade_duration_bars": 10, "sharpe_ratio": 2.0}
    cost2x       = {"sharpe_ratio": 0.5}   # dropped 75% -> > 30% threshold
    result = hard_rejection_checks(base_metrics, cost_sensitivity_metrics=cost2x, walk_forward_run=True)
    assert result["hard_rejected"]
    assert "cost_sensitive" in result["rejections"]


def test_hard_rejection_wf_not_run():
    metrics = {"avg_trade_duration_bars": 10, "sharpe_ratio": 1.5}
    result  = hard_rejection_checks(metrics, walk_forward_run=False)
    assert result["hard_rejected"]
    assert "wf_not_run" in result["rejections"]


def test_no_hard_rejection_clean():
    metrics = {"avg_trade_duration_bars": 10, "sharpe_ratio": 1.5}
    train_dist = {"bull": 0.40, "bear": 0.30, "sideways": 0.30}
    test_dist  = {"bull": 0.38, "bear": 0.32, "sideways": 0.30}
    cost2x     = {"sharpe_ratio": 1.2}  # dropped only 20%
    result = hard_rejection_checks(metrics, train_dist, test_dist, cost2x, walk_forward_run=True)
    assert not result["hard_rejected"]


# ---- multi_split_verdict ----

def _good_train():
    return {"sharpe_ratio": 1.0, "profit_factor": 1.3, "total_trades": 100}

def _good_val():
    return {"sharpe_ratio": 0.8, "profit_factor": 1.2, "total_trades": 50}

def _good_test():
    return {
        "annualized_return": 0.25, "sharpe_ratio": 1.3, "max_drawdown": -0.18,
        "profit_factor": 1.5, "total_trades": 70, "win_rate": 0.52,
    }


def test_verdict_approved(base_config):
    mv = multi_split_verdict(
        _good_train(), _good_val(), _good_test(), base_config,
        wf_results={"mean_sharpe": 0.8, "pct_positive": 0.80},
        hard_checks={"hard_rejected": False, "rejections": {}},
    )
    assert mv["verdict"] == "APPROVED"


def test_verdict_hard_rejected(base_config):
    mv = multi_split_verdict(
        _good_train(), _good_val(), _good_test(), base_config,
        hard_checks={"hard_rejected": True, "rejections": {"wf_not_run": "..."}},
    )
    assert mv["verdict"] == "HARD_REJECTED"


def test_verdict_revise_partial_test(base_config):
    test_metrics = {
        "annualized_return": 0.05,   # below 10% threshold
        "sharpe_ratio": 1.3, "max_drawdown": -0.18,
        "profit_factor": 1.5, "total_trades": 70, "win_rate": 0.52,
    }
    mv = multi_split_verdict(
        _good_train(), _good_val(), test_metrics, base_config,
        hard_checks={"hard_rejected": False, "rejections": {}},
    )
    assert mv["verdict"] in ("REVISE", "REJECTED")


def test_verdict_overfit(base_config):
    bad_train = {"sharpe_ratio": -0.5, "profit_factor": 0.8, "total_trades": 30}
    mv = multi_split_verdict(
        bad_train, _good_val(), _good_test(), base_config,
        hard_checks={"hard_rejected": False, "rejections": {}},
    )
    assert mv["verdict"] == "OVERFIT"
