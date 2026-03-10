"""
evaluation/verdict.py - Multi-split verdict logic.
"""

from typing import Any, Dict, Optional

from trade2.evaluation.acceptance import passes_criteria, passes_walk_forward


def multi_split_verdict(
    train_metrics: Dict[str, Any],
    val_metrics:   Dict[str, Any],
    test_metrics:  Dict[str, Any],
    config:        Dict[str, Any],
    wf_results:    Dict[str, Any] = None,
    hard_checks:   Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Strict multi-split verdict. ALL splits must meet thresholds.
    A strategy passing test but failing train is likely overfit.

    Returns dict with per-split results and final verdict string.
    """
    train_pass = passes_criteria(train_metrics, config, "train")
    val_pass   = passes_criteria(val_metrics,   config, "val")
    test_pass  = passes_criteria(test_metrics,  config, "test")
    wf_pass    = passes_walk_forward(wf_results, config) if wf_results else {}

    train_ok = all(train_pass.values())
    val_ok   = all(val_pass.values())
    test_ok  = all(test_pass.values())
    wf_ok    = wf_pass.get("available", False) and all(
        v for k, v in wf_pass.items() if k != "available"
    )
    wf_run   = wf_pass.get("available", False)

    if hard_checks is None:
        hard_checks = {}
    hard_rejected = hard_checks.get("hard_rejected", False)

    if hard_rejected:
        final = "HARD_REJECTED"
    elif test_ok and train_ok and val_ok and (wf_ok or not wf_run):
        final = "APPROVED"
    elif test_ok and not train_ok:
        final = "OVERFIT"
    elif test_ok and not val_ok:
        final = "UNSTABLE"
    elif not any(test_pass.values()):
        final = "REJECTED"
    else:
        final = "REVISE"

    return {
        "verdict":    final,
        "train_pass": train_pass,
        "val_pass":   val_pass,
        "test_pass":  test_pass,
        "wf_pass":    wf_pass,
        "train_ok":   train_ok,
        "val_ok":     val_ok,
        "test_ok":    test_ok,
        "wf_ok":      wf_ok,
        "hard_checks": hard_checks,
        "flags": {
            "test_without_train": test_ok and not train_ok,
            "test_without_val":   test_ok and not val_ok,
            "wf_not_run":         not wf_run,
            "hard_rejected":      hard_rejected,
        },
    }


def verdict(metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Single-split verdict. Prefer multi_split_verdict for production use."""
    criteria  = passes_criteria(metrics, config, "test")
    all_pass  = all(criteria.values())
    none_pass = not any(criteria.values())
    if all_pass:   return "APPROVED"
    if none_pass:  return "REJECTED"
    return "REVISE"
