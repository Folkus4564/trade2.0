"""
Module: pipeline.py  (v2)
Purpose: Multi-timeframe XAUUSD strategy pipeline
         - 1H HMM regime detection (3 states, 7 features)
         - 5M SMC entry signals (OB, FVG, sweeps, confluence)
         - Consistent train/val/test splits across both timeframes

Usage:
    python src_v2/pipeline.py
    python src_v2/pipeline.py --optimize --trials 100
"""

import argparse
import json
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src_v2.data.loader import load_split_tf
from src_v2.data.features import add_1h_features, get_hmm_feature_matrix, add_5m_features
from src_v2.models.hmm_model import XAUUSDRegimeModel
from src_v2.models.signal_generator import (
    forward_fill_1h_regime,
    generate_signals,
    compute_stops,
)
from src_v2.backtesting.engine import run_backtest

STRATEGY_NAME = "xauusd_mtf_hmm1h_smc5m"
REPORTS_DIR   = ROOT / "reports_v2"
REPORTS_DIR.mkdir(exist_ok=True)


def load_v2_config() -> tuple:
    """
    Load config.yaml and build the v2 params dict.
    Returns (cfg, params) where cfg is the raw dict and params mirrors DEFAULT_PARAMS.
    Raises KeyError with a clear message if any expected key is missing.
    """
    config_path = ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def _get(path, cfg=cfg):
        keys = path.split(".")
        node = cfg
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(
                    f"[load_v2_config] Missing config key: '{path}' "
                    f"(failed at '{k}'). Check config.yaml."
                )
            node = node[k]
        return node

    params = {
        "hmm_states":             _get("hmm.n_states"),
        "hmm_min_prob":           _get("hmm.min_prob_hard"),
        "regime_persistence_bars": _get("regime.persistence_bars"),
        "adx_threshold":          _get("regime.adx_threshold"),
        "atr_stop_mult":          _get("risk.atr_stop_mult"),
        "atr_tp_mult":            _get("risk.atr_tp_mult"),
        "require_smc_confluence": _get("smc.require_confluence"),
        "require_pin_bar":        _get("smc.require_pin_bar"),
        "dc_period":              _get("features.dc_period"),
    }
    return cfg, params


def run_pipeline(params: dict = None) -> dict:
    """
    Full v2 pipeline:
    1. Load 1H + 5M data with consistent splits
    2. Build 1H features, train HMM on 1H train
    3. Build 5M features
    4. Forward-fill 1H regime onto 5M bars
    5. Generate 5M SMC signals with regime filter
    6. Backtest on 5M bars
    7. Save verdict
    """
    cfg, default_params = load_v2_config()
    np.random.seed(cfg["hmm"]["random_seed"])
    p = {**default_params, **(params or {})}

    print(f"\n{'='*60}")
    print(f"  XAUUSD MTF STRATEGY v2  |  1H HMM + 5M SMC")
    print(f"{'='*60}")
    print(f"  HMM states : {p['hmm_states']}  |  min_prob: {p['hmm_min_prob']}")
    print(f"  Persistence: {p['regime_persistence_bars']} 1H bars")
    print(f"  Confluence : {p['require_smc_confluence']}  |  Pin bar: {p['require_pin_bar']}")
    print(f"  ATR SL/TP  : {p['atr_stop_mult']}x / {p['atr_tp_mult']}x")
    print(f"{'='*60}\n")

    # ---- 1. Load data ----
    print("[pipeline_v2] Loading 1H data...")
    train_1h, val_1h, test_1h = load_split_tf("1H", config=cfg)

    print("[pipeline_v2] Loading 5M data...")
    train_5m, val_5m, test_5m = load_split_tf("5M", config=cfg)

    print(f"\n  1H  train={len(train_1h)} | val={len(val_1h)} | test={len(test_1h)} bars")
    print(f"  5M  train={len(train_5m)} | val={len(val_5m)} | test={len(test_5m)} bars\n")

    # ---- 2. 1H features + HMM ----
    print("[pipeline_v2] Engineering 1H features...")
    train_1h_feat = add_1h_features(train_1h)
    val_1h_feat   = add_1h_features(val_1h)
    test_1h_feat  = add_1h_features(test_1h)

    print("[pipeline_v2] Training HMM on 1H train data...")
    X_train_1h, idx_train_1h = get_hmm_feature_matrix(train_1h_feat)

    hmm = XAUUSDRegimeModel(n_states=p["hmm_states"], random_seed=cfg["hmm"]["random_seed"])
    hmm.fit(X_train_1h)
    hmm.save("hmm_regime_model_v2")
    hmm.summary(X_train_1h)

    # Validate state distribution
    dist = hmm.state_distribution(X_train_1h)
    print(f"\n[pipeline_v2] Regime distribution on train:")
    for label, pct in dist.items():
        print(f"  {label:>10}: {pct*100:.1f}%")

    if p["hmm_states"] == 3:
        sideways_pct = dist.get("sideways", 0)
        if sideways_pct < 0.10:
            print("[pipeline_v2] WARNING: sideways < 10% of bars, consider using 2-state HMM")

    # ---- 3. 5M features ----
    print("\n[pipeline_v2] Engineering 5M features (SMC)...")
    train_5m_feat = add_5m_features(train_5m, dc_period=p["dc_period"], smc_config=cfg.get("smc_5m"))
    val_5m_feat   = add_5m_features(val_5m,   dc_period=p["dc_period"], smc_config=cfg.get("smc_5m"))
    test_5m_feat  = add_5m_features(test_5m,  dc_period=p["dc_period"], smc_config=cfg.get("smc_5m"))

    for col in ["ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish", "sweep_low", "sweep_high"]:
        if col in train_5m_feat.columns:
            print(f"  [SMC] {col}: {int(train_5m_feat[col].sum())} active bars (train)")

    # ---- 4. Forward-fill 1H regime onto 5M ----
    print("\n[pipeline_v2] Forward-filling 1H regime onto 5M bars...")

    def apply_regime(feat_1h, feat_5m, hmm_model):
        X, idx = get_hmm_feature_matrix(feat_1h)
        labels    = hmm_model.regime_labels(X)
        bull_prob = hmm_model.bull_probability(X)
        bear_prob = hmm_model.bear_probability(X)
        return forward_fill_1h_regime(feat_5m, labels, bull_prob, bear_prob, idx)

    train_5m_regime = apply_regime(train_1h_feat, train_5m_feat, hmm)
    val_5m_regime   = apply_regime(val_1h_feat,   val_5m_feat,   hmm)
    test_5m_regime  = apply_regime(test_1h_feat,  test_5m_feat,  hmm)

    # ---- 5. Generate 5M signals ----
    print("[pipeline_v2] Generating 5M signals...")

    def apply_signals(df):
        sig = generate_signals(
            df,
            adx_threshold            = p["adx_threshold"],
            hmm_min_prob             = p["hmm_min_prob"],
            regime_persistence_bars  = p["regime_persistence_bars"],
            require_smc_confluence   = p["require_smc_confluence"],
            require_pin_bar          = p["require_pin_bar"],
        )
        return compute_stops(sig, p["atr_stop_mult"], p["atr_tp_mult"])

    train_sig = apply_signals(train_5m_regime)
    val_sig   = apply_signals(val_5m_regime)
    test_sig  = apply_signals(test_5m_regime)

    print(f"  Train: long={int(train_sig['signal_long'].sum())} | short={int(train_sig['signal_short'].sum())}")
    print(f"  Test : long={int(test_sig['signal_long'].sum())}  | short={int(test_sig['signal_short'].sum())}")

    # ---- 6. Backtests ----
    print("\n[pipeline_v2] Running TRAIN backtest...")
    train_metrics, _ = run_backtest(train_sig, STRATEGY_NAME, "train", config=cfg)

    print("\n[pipeline_v2] Running TEST backtest (out-of-sample)...")
    test_metrics, _  = run_backtest(test_sig, STRATEGY_NAME, "test", config=cfg)

    # ---- 7. Verdict ----
    from src.backtesting.metrics import passes_criteria, verdict

    final_verdict = verdict(test_metrics)
    criteria      = passes_criteria(test_metrics)

    results = {
        "strategy_name":  STRATEGY_NAME,
        "verdict":        final_verdict,
        "date":           str(date.today()),
        "params":         p,
        "train_metrics":  train_metrics,
        "test_metrics":   test_metrics,
        "pass_criteria":  criteria,
    }

    verdict_path = REPORTS_DIR / "final_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(test_metrics, final_verdict)
    print(f"[pipeline_v2] Verdict saved to {verdict_path}")

    return results


def _print_summary(metrics: dict, v: str) -> None:
    SEP = "=" * 55
    print(f"\n{SEP}")
    print(f"  XAUUSD MTF v2 FINAL RESULTS  (5M test period)")
    print(f"{SEP}")
    print(f"  Annualized Return : {metrics.get('annualized_return', 0)*100:>8.2f}%  (target: >=20%)")
    print(f"  Sharpe Ratio      : {metrics.get('sharpe_ratio', 0):>8.3f}  (target: >=1.0)")
    print(f"  Sortino Ratio     : {metrics.get('sortino_ratio', 0):>8.3f}")
    print(f"  Max Drawdown      : {metrics.get('max_drawdown', 0)*100:>8.2f}%  (target: >=-35%)")
    print(f"  Calmar Ratio      : {metrics.get('calmar_ratio', 0):>8.3f}")
    print(f"  Total Trades      : {metrics.get('total_trades', 'N/A'):>8}")
    print(f"  Win Rate          : {metrics.get('win_rate', 0)*100:>8.2f}%")
    print(f"  Profit Factor     : {metrics.get('profit_factor', 0):>8.3f}")
    if "benchmark_return" in metrics:
        print(f"  vs Buy & Hold     : {metrics.get('alpha_vs_benchmark', 0)*100:>+8.2f}%")
    print(f"{SEP}")
    print(f"  Verdict: {v}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD MTF v2 Pipeline")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna optimization")
    parser.add_argument("--trials",   type=int, default=100, help="Optuna trial count")
    args = parser.parse_args()

    results = run_pipeline()
    sys.exit(0 if results["verdict"] == "APPROVED" else 1)
