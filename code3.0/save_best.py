"""
save_best.py - Save best_4h_43pct strategy to artefacts/approved_strategies/.
Runs the pipeline (no WF, no optimize) with the best config, then exports regardless of verdict.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from trade2.config.loader import load_config
from trade2.app.run_pipeline import run_pipeline
from trade2.export.exporter import export_approved_strategy

config = load_config(ROOT / "configs" / "base.yaml",
                     ROOT / "configs" / "best_4h_43pct.yaml")

print("[save_best] Running pipeline with best_4h_43pct config ...")
results = run_pipeline(
    config          = config,
    retrain_model   = True,
    walk_forward    = False,
    optimize        = False,
    export_approved = False,
    legacy_signals  = True,
)

tm = results.get("test_metrics", {})
print(f"\n[save_best] Test: Return={tm.get('annualized_return',0)*100:.1f}%  "
      f"Sharpe={tm.get('sharpe_ratio',0):.3f}  "
      f"MaxDD={tm.get('max_drawdown',0)*100:.1f}%  "
      f"Trades={tm.get('total_trades',0)}")
print(f"[save_best] Verdict: {results.get('verdict')}")

artefacts_dir = ROOT / "artefacts"
model_path    = artefacts_dir / "models" / "hmm_4h_2states.pkl"

export_dir = export_approved_strategy(
    results        = results,
    config         = config,
    artefacts_dir  = artefacts_dir,
    model_path     = model_path,
    strategy_name  = "best_4h_43pct_2state",
)

print(f"\n[save_best] Saved to: {export_dir}")
