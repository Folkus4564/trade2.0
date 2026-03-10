"""
Module: experiment.py
Purpose: Log every pipeline run as a timestamped JSON experiment for reproducibility
Author: Auto-generated
Date: 2026-03-10
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def _git_hash() -> str:
    """Return short git commit hash, or 'unknown' if git not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class ExperimentLogger:
    """
    Logs each pipeline run as a JSON file in experiments/.
    File name: experiments/YYYYMMDD_HHMMSS_<verdict>.json
    """

    def __init__(self):
        self.run_id   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.git_hash = _git_hash()
        self._data: Dict[str, Any] = {}

    def log(
        self,
        params:        Dict[str, Any],
        config:        Dict[str, Any],
        train_metrics: Dict[str, Any],
        val_metrics:   Dict[str, Any],
        test_metrics:  Dict[str, Any],
        verdict:       str,
        wf_results:    Optional[Dict[str, Any]] = None,
        hard_checks:   Optional[Dict[str, Any]] = None,
        extra:         Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save experiment to disk.

        Returns:
            Path to saved JSON file.
        """
        self._data = {
            "run_id":        self.run_id,
            "git_hash":      self.git_hash,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "verdict":       verdict,
            "params":        params,
            "config":        config,
            "metrics": {
                "train": train_metrics,
                "val":   val_metrics,
                "test":  test_metrics,
            },
            "walk_forward": wf_results,
            "hard_checks":  hard_checks,
        }
        if extra:
            self._data.update(extra)

        out_path = EXPERIMENTS_DIR / f"{self.run_id}_{verdict}.json"
        out_path.write_text(json.dumps(self._data, indent=2))
        print(f"[experiment] Logged run {self.run_id} -> {out_path.name}")
        return out_path


def list_experiments() -> List[Dict[str, Any]]:
    """Return summary list of all past experiments (sorted by newest first)."""
    rows = []
    for fp in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(fp.read_text())
            rows.append({
                "run_id":   data.get("run_id"),
                "verdict":  data.get("verdict"),
                "timestamp": data.get("timestamp_utc"),
                "test_sharpe": data.get("metrics", {}).get("test", {}).get("sharpe_ratio"),
                "test_return": data.get("metrics", {}).get("test", {}).get("annualized_return"),
                "git_hash": data.get("git_hash"),
                "file":     fp.name,
            })
        except Exception:
            continue
    return rows


def best_experiment(metric: str = "test_sharpe") -> Optional[Dict[str, Any]]:
    """Retrieve the best experiment by a summary metric (e.g. 'test_sharpe')."""
    exps = list_experiments()
    valid = [e for e in exps if e.get(metric) is not None]
    if not valid:
        return None
    return max(valid, key=lambda e: e[metric])
