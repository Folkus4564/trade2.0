"""
experiment/logger.py - Log every pipeline run as a timestamped JSON.
experiments_dir passed explicitly (no module-level mkdir).
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git_hash(cwd: Path = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(cwd or Path.cwd()),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class ExperimentLogger:
    """
    Logs each pipeline run as experiments/<run_id>_<verdict>.json.
    experiments_dir must be passed explicitly (no side effects at import).
    """

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = Path(experiments_dir)
        self.run_id   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.git_hash = _git_hash(self.experiments_dir.parent)

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
        """Save experiment to disk. Returns path to saved JSON."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        data = {
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
            data.update(extra)

        out_path = self.experiments_dir / f"{self.run_id}_{verdict}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"[experiment] Logged run {self.run_id} -> {out_path.name}")
        return out_path


def list_experiments(experiments_dir: Path) -> List[Dict[str, Any]]:
    """Return summary list of all past experiments (newest first)."""
    rows = []
    for fp in sorted(Path(experiments_dir).glob("*.json"), reverse=True):
        try:
            data = json.loads(fp.read_text())
            rows.append({
                "run_id":      data.get("run_id"),
                "verdict":     data.get("verdict"),
                "timestamp":   data.get("timestamp_utc"),
                "test_sharpe": data.get("metrics", {}).get("test", {}).get("sharpe_ratio"),
                "test_return": data.get("metrics", {}).get("test", {}).get("annualized_return"),
                "git_hash":    data.get("git_hash"),
                "file":        fp.name,
            })
        except Exception:
            continue
    return rows
