"""
export/exporter.py - Export approved strategies to artefacts/approved_strategies/.
artefacts_dir passed explicitly (no module-level side effects).
"""

import json
import shutil
import yaml
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional


def export_approved_strategy(
    results:       Dict[str, Any],
    config:        Dict[str, Any],
    artefacts_dir: Path,
    model_path:    Optional[Path] = None,
    strategy_name: str = "xauusd_mtf_hmm_smc",
) -> Path:
    """
    Export approved strategy to artefacts/approved_strategies/<name>_<date>/.

    Creates:
        config.yaml          - config snapshot
        metrics.json         - all split metrics
        training_summary.md  - human-readable summary
        model.pkl            - trained HMM model copy (if model_path provided)

    Args:
        results:       Full results dict from pipeline
        config:        Config dict snapshot
        artefacts_dir: Root artefacts directory
        model_path:    Path to trained HMM .pkl (optional)
        strategy_name: Base name for the export folder

    Returns:
        Path to created export directory.
    """
    today      = date.today().strftime("%Y_%m_%d")
    export_dir = Path(artefacts_dir) / "approved_strategies" / f"{strategy_name}_{today}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # config.yaml snapshot
    with open(export_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # metrics.json
    metrics_data = {
        "strategy_name": results.get("strategy_name"),
        "date":          results.get("date"),
        "verdict":       results.get("verdict"),
        "params":        results.get("params"),
        "train":         results.get("train_metrics"),
        "val":           results.get("val_metrics"),
        "test":          results.get("test_metrics"),
        "walk_forward":  results.get("walk_forward"),
        "hard_checks":   results.get("verdict_detail", {}).get("hard_checks"),
    }
    with open(export_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # model.pkl copy
    if model_path and Path(model_path).exists():
        shutil.copy(model_path, export_dir / "model.pkl")

    # training_summary.md
    _write_summary(export_dir, results)

    print(f"[export] Strategy exported to {export_dir}")
    return export_dir


def _write_summary(export_dir: Path, results: Dict[str, Any]) -> None:
    test  = results.get("test_metrics", {})
    train = results.get("train_metrics", {})
    val   = results.get("val_metrics",   {})

    def pct(v): return f"{v*100:.2f}%" if v is not None else "N/A"
    def num(v): return f"{v:.3f}"      if v is not None else "N/A"

    lines = [
        f"# Strategy Export: {results.get('strategy_name', 'unknown')}",
        f"",
        f"**Export date**: {results.get('date')}",
        f"**Verdict**: {results.get('verdict')}",
        f"",
        f"## Performance Summary",
        f"",
        f"| Metric | Train | Val | Test |",
        f"|--------|-------|-----|------|",
        f"| Annualized Return | {pct(train.get('annualized_return'))} | {pct(val.get('annualized_return'))} | {pct(test.get('annualized_return'))} |",
        f"| Sharpe Ratio | {num(train.get('sharpe_ratio'))} | {num(val.get('sharpe_ratio'))} | {num(test.get('sharpe_ratio'))} |",
        f"| Max Drawdown | {pct(train.get('max_drawdown'))} | {pct(val.get('max_drawdown'))} | {pct(test.get('max_drawdown'))} |",
        f"| Profit Factor | {num(train.get('profit_factor'))} | {num(val.get('profit_factor'))} | {num(test.get('profit_factor'))} |",
        f"| Total Trades | {train.get('total_trades','N/A')} | {val.get('total_trades','N/A')} | {test.get('total_trades','N/A')} |",
        f"| Win Rate | {pct(train.get('win_rate'))} | {pct(val.get('win_rate'))} | {pct(test.get('win_rate'))} |",
        f"",
        f"## Parameters",
        f"",
        f"```json",
        json.dumps(results.get("params", {}), indent=2),
        f"```",
    ]

    wf = results.get("walk_forward")
    if wf and wf.get("available"):
        lines += [
            f"",
            f"## Walk-Forward Results",
            f"",
            f"- Mean Sharpe: {num(wf.get('mean_sharpe'))}",
            f"- Positive Windows: {pct(wf.get('pct_positive'))}",
            f"- Windows: {wf.get('n_windows')}",
        ]

    (export_dir / "training_summary.md").write_text("\n".join(lines))
