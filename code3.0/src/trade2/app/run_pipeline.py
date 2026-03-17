"""
app/run_pipeline.py - CLI entry point and thin orchestrator.
Replaces both src/pipeline.py and src_v2/pipeline.py.

Usage:
    trade2 --config configs/xauusd_mtf.yaml --retrain-model
    trade2 --skip-walk-forward
    trade2 --optimize --trials 200
    trade2 --retrain-model --export-approved
"""

import argparse
import hashlib
import json
import pickle
import sys
import numpy as np
from datetime import date
from pathlib import Path

from trade2.config.loader import load_config
from trade2.data.splits import load_split_tf
from trade2.data.loader import dataset_version
from trade2.data.validation import audit_missing_bars
from trade2.features.builder import add_1h_features, add_5m_features
from trade2.features.hmm_features import get_hmm_feature_matrix
from trade2.models.hmm import XAUUSDRegimeModel
from trade2.signals.regime import forward_fill_1h_regime
from trade2.signals.generator import generate_signals, compute_stops, compute_stops_regime_aware
from trade2.signals.router import route_signals
from trade2.backtesting.engine import run_backtest, run_backtest_2x_costs, run_walk_forward
from trade2.evaluation.hard_rejection import hard_rejection_checks
from trade2.evaluation.verdict import multi_split_verdict
from trade2.experiment.logger import ExperimentLogger
from trade2.export.exporter import export_approved_strategy
from trade2.optimization.optimizer import run_optimization

PROJECT_ROOT = Path(__file__).parents[3]  # code3.0/
DATA_ROOT    = Path(__file__).parents[4]  # trade2.0/ (where data/ lives)


def _feat_cache_key(config: dict, tf: str, split: str) -> str:
    """Short hash of feature-relevant config keys + tf + split."""
    keys = {
        "tf": tf, "split": split,
        "features": config.get("features", {}),
        "smc":      config.get("smc",      {}),
        "smc_5m":   config.get("smc_5m",   {}),
        "smc_luxalgo":    config.get("smc_luxalgo",    {}),
        "smc_luxalgo_5m": config.get("smc_luxalgo_5m", {}),
        "strategies_cdc": config.get("strategies", {}).get("cdc", {}),
    }
    return hashlib.md5(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _load_features_cached(df, tf, split, config, dirs, add_fn, **kwargs):
    """Call add_fn(df, config, **kwargs), optionally caching result to disk."""
    if not config.get("pipeline", {}).get("cache_features", False):
        return add_fn(df, config, **kwargs)
    cache_dir = dirs["root"] / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key  = _feat_cache_key(config, tf, split)
    path = cache_dir / f"{tf}_{split}_{key}.pkl"
    if path.exists():
        print(f"  [cache] Loading {tf} {split} features from cache")
        with open(path, "rb") as f:
            return pickle.load(f)
    result = add_fn(df, config, **kwargs)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"  [cache] Saved {tf} {split} features to cache")
    return result


def _resolve_artefact_dirs(config: dict) -> dict:
    """Resolve artefact directory paths from config, relative to project root."""
    art_cfg = config.get("artefacts", {})
    root    = PROJECT_ROOT / art_cfg.get("root", "artefacts")
    return {
        "root":       root,
        "models":     PROJECT_ROOT / art_cfg.get("models",      "artefacts/models"),
        "backtests":  PROJECT_ROOT / art_cfg.get("backtests",   "artefacts/backtests"),
        "reports":    PROJECT_ROOT / art_cfg.get("reports",     "artefacts/reports"),
        "experiments": PROJECT_ROOT / art_cfg.get("experiments","artefacts/experiments"),
        "approved":   PROJECT_ROOT / art_cfg.get("approved",    "artefacts/approved_strategies"),
    }


def _regime_distribution(labels) -> dict:
    import pandas as pd
    s = pd.Series(labels.values if hasattr(labels, "values") else list(labels))
    counts = s.value_counts(normalize=True)
    return {regime: round(float(frac), 4) for regime, frac in counts.items()}


def _build_params(config: dict) -> dict:
    feat = config["features"]
    smc  = config["smc"]
    risk = config["risk"]
    hmm  = config["hmm"]
    reg  = config["regime"]
    return {
        "hma_period":              feat["hma_period"],
        "ema_period":              feat["ema_period"],
        "atr_period":              feat["atr_period"],
        "rsi_period":              feat["rsi_period"],
        "adx_period":              feat["adx_period"],
        "dc_period":               feat["dc_period"],
        "adx_threshold":           reg["adx_threshold"],
        "hmm_min_prob":            hmm["min_prob_hard"],
        "hmm_states":              hmm["n_states"],
        "regime_persistence_bars": reg["persistence_bars"],
        "atr_stop_mult":           risk["atr_stop_mult"],
        "atr_tp_mult":             risk["atr_tp_mult"],
        "require_smc_confluence":  smc["require_confluence"],
        "require_pin_bar":         smc["require_pin_bar"],
    }


def _log_signal_stats(sig_df, config: dict) -> None:
    """Log probabilistic gating diagnostic stats."""
    hmm_cfg = config["hmm"]
    reg_cfg = config["regime"]
    min_confidence = hmm_cfg["min_confidence"]
    cooldown       = reg_cfg["transition_cooldown_bars"]

    # Fraction filtered by min_confidence (max-entropy bars)
    if "bull_prob" in sig_df.columns and "bear_prob" in sig_df.columns:
        sideways_col = sig_df["sideways_prob"] if "sideways_prob" in sig_df.columns \
                       else (1.0 - sig_df["bull_prob"] - sig_df["bear_prob"]).clip(lower=0.0)
        max_prob = sig_df[["bull_prob", "bear_prob"]].assign(sideways_prob=sideways_col).max(axis=1)
        frac_low_confidence = float((max_prob < min_confidence).mean())
        print(f"  [prob-gate] bars filtered by min_confidence ({min_confidence}): {frac_low_confidence*100:.1f}%")

    # Fraction filtered by transition cooldown
    if cooldown > 0 and "regime" in sig_df.columns:
        from trade2.signals.generator import _is_5m_data
        freq_mult     = 12 if _is_5m_data(sig_df) else 1
        cooldown_bars = cooldown * freq_mult
        regime_changed = sig_df["regime"] != sig_df["regime"].shift(1)
        in_cooldown    = regime_changed.rolling(cooldown_bars, min_periods=1).sum() > 0
        frac_cooldown  = float(in_cooldown.mean())
        print(f"  [prob-gate] bars filtered by cooldown ({cooldown}h): {frac_cooldown*100:.1f}%")

    # Mean position size on signal bars
    any_signal = (sig_df["signal_long"] == 1) | (sig_df["signal_short"] == 1)
    if any_signal.any():
        mean_size_long  = sig_df.loc[sig_df["signal_long"]  == 1, "position_size_long"].mean()  if sig_df["signal_long"].any()  else 0.0
        mean_size_short = sig_df.loc[sig_df["signal_short"] == 1, "position_size_short"].mean() if sig_df["signal_short"].any() else 0.0
        print(f"  [prob-gate] mean position size: long={mean_size_long:.3f} | short={mean_size_short:.3f}")


def run_pipeline(
    config: dict,
    params: dict = None,
    walk_forward: bool = True,
    retrain_model: bool = False,
    export_approved: bool = False,
    optimize: bool = False,
    n_trials: int = 100,
    legacy_signals: bool = False,
    optuna_target: str = "val_sharpe",
    return_trades: bool = False,
) -> dict:
    """
    Full research pipeline:
    1. Load data (1H always; 5M if multi_tf mode)
    2. Build features
    3. Train/load HMM
    4. Forward-fill regime (multi_tf only)
    5. Generate signals
    6. Backtest all splits + 2x cost sensitivity
    7. Walk-forward validation
    8. Hard rejection checks
    9. Multi-split verdict
    10. Experiment logging
    11. Export if approved + flag set
    """
    p      = {**_build_params(config), **(params or {})}
    dirs   = _resolve_artefact_dirs(config)
    mode   = config.get("strategy", {}).get("mode", "multi_tf")
    strategy_name = config.get("strategy", {}).get("name", "xauusd_mtf_hmm_smc")

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    np.random.seed(config.get("hmm", {}).get("random_seed", 42))

    print(f"\n{'='*60}")
    print(f"  XAUUSD STRATEGY PIPELINE  |  mode={mode}")
    print(f"{'='*60}")
    print(f"  Strategy    : {strategy_name}")
    print(f"  HMM prob    : {p['hmm_min_prob']}  |  states: {p['hmm_states']}")
    print(f"  ATR SL/TP   : {p['atr_stop_mult']}x / {p['atr_tp_mult']}x")
    print(f"  Confluence  : {p['require_smc_confluence']}")
    print(f"{'='*60}\n")

    # ---- 1. Load data ----
    regime_tf = config.get("strategy", {}).get("regime_timeframe", "1H")
    signal_tf = config.get("strategy", {}).get("signal_timeframe", "5M")
    _TF_TO_FREQ = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"}

    print(f"[pipeline] Loading regime TF ({regime_tf}) data...")
    train_reg, val_reg, test_reg = load_split_tf(regime_tf, config)

    raw_1h_path = DATA_ROOT / config.get("data", {}).get("raw_1h_csv", "data/raw/XAUUSD_1H_2019_2025.csv")

    # Resolve raw CSV for regime TF (needed for walk-forward which reloads its own data)
    _TF_TO_RAW_KEY = {"1H": "raw_1h_csv", "5M": "raw_5m_csv", "4H": "raw_4h_csv"}
    _TF_DEFAULTS   = {"1H": "data/raw/XAUUSD_1H_2019_2025.csv",
                      "5M": "data/raw/XAUUSD_5M_2019_2025.csv",
                      "4H": "data/raw/XAUUSD_4H_2019_2025.csv"}
    raw_regime_path = DATA_ROOT / config.get("data", {}).get(
        _TF_TO_RAW_KEY.get(regime_tf, "raw_1h_csv"),
        _TF_DEFAULTS.get(regime_tf, "data/raw/XAUUSD_1H_2019_2025.csv"),
    )
    if not raw_regime_path.exists():
        print(f"[pipeline] WARNING: regime raw CSV not found at {raw_regime_path}, falling back to 1H")
        raw_regime_path = raw_1h_path
    gap_audit  = audit_missing_bars(train_reg, regime_tf.lower(), config)
    data_ver   = dataset_version(raw_1h_path) if raw_1h_path.exists() else {}
    print(f"  [data] Missing bars: {gap_audit['missing_bars']} ({gap_audit['missing_pct']}%)")

    if mode == "multi_tf":
        print(f"[pipeline] Loading signal TF ({signal_tf}) data...")
        train_sig_raw, val_sig_raw, test_sig_raw = load_split_tf(signal_tf, config)
        print(f"  {regime_tf} train={len(train_reg)} | val={len(val_reg)} | test={len(test_reg)}")
        print(f"  {signal_tf} train={len(train_sig_raw)} | val={len(val_sig_raw)} | test={len(test_sig_raw)}")
    else:
        print(f"  {regime_tf} train={len(train_reg)} | val={len(val_reg)} | test={len(test_reg)}")

    # ---- 2. Build features (optionally cached) ----
    print(f"[pipeline] Engineering {regime_tf} regime features...")
    train_reg_feat = _load_features_cached(train_reg, regime_tf, "train", config, dirs, add_1h_features)
    val_reg_feat   = _load_features_cached(val_reg,   regime_tf, "val",   config, dirs, add_1h_features)
    test_reg_feat  = _load_features_cached(test_reg,  regime_tf, "test",  config, dirs, add_1h_features)

    if mode == "multi_tf":
        print(f"[pipeline] Engineering {signal_tf} signal features (SMC)...")
        train_sig_feat = _load_features_cached(train_sig_raw, signal_tf, "train", config, dirs, add_5m_features, dc_period=p["dc_period"])
        val_sig_feat   = _load_features_cached(val_sig_raw,   signal_tf, "val",   config, dirs, add_5m_features, dc_period=p["dc_period"])
        test_sig_feat  = _load_features_cached(test_sig_raw,  signal_tf, "test",  config, dirs, add_5m_features, dc_period=p["dc_period"])
        for col in ["ob_bullish","ob_bearish","fvg_bullish","fvg_bearish","sweep_low","sweep_high"]:
            if col in train_sig_feat.columns:
                print(f"  [SMC {signal_tf}] {col}: {int(train_sig_feat[col].sum())} active bars (train)")
    else:
        for col in ["ob_bullish","ob_bearish","fvg_bullish","fvg_bearish","sweep_low","sweep_high"]:
            if col in train_reg_feat.columns:
                print(f"  [SMC {regime_tf}] {col}: {int(train_reg_feat[col].sum())} active bars (train)")

    # ---- 3. Train / load HMM ----
    # Model filename encodes regime_tf + n_states so each combo has its own file.
    # e.g. hmm_1h_3states.pkl, hmm_4h_3states.pkl, hmm_1h_2states.pkl
    _model_tf_key = regime_tf.lower().replace(" ", "")   # "1H" -> "1h", "4H" -> "4h"
    model_path = dirs["models"] / f"hmm_{_model_tf_key}_{p['hmm_states']}states.pkl"
    if retrain_model or not model_path.exists():
        print("[pipeline] Training HMM regime model...")
        X_train, _ = get_hmm_feature_matrix(train_reg_feat, config)
        hmm = XAUUSDRegimeModel(
            n_states    = p["hmm_states"],
            random_seed = config.get("hmm", {}).get("random_seed", 42),
        )
        hmm.fit(X_train)
        hmm.save(model_path)
        hmm.summary(X_train)
    else:
        print(f"[pipeline] Loading existing HMM model from {model_path}")
        hmm = XAUUSDRegimeModel.load(model_path)

    # ---- 4. Get regime for all regime-TF splits ----
    def _get_regime(feat_reg):
        X, idx = get_hmm_feature_matrix(feat_reg, config)
        return (hmm.regime_labels(X), hmm.bull_probability(X), hmm.bear_probability(X),
                hmm.sideways_probability(X), X, idx)

    train_labels, train_bull, train_bear, train_sideways, X_train_reg, idx_train_reg = _get_regime(train_reg_feat)
    val_labels,   val_bull,   val_bear,   val_sideways,   X_val_reg,   idx_val_reg   = _get_regime(val_reg_feat)
    test_labels,  test_bull,  test_bear,  test_sideways,  X_test_reg,  idx_test_reg  = _get_regime(test_reg_feat)

    train_regime_dist = _regime_distribution(train_labels)
    test_regime_dist  = _regime_distribution(test_labels)
    print(f"  [regime] Train: {train_regime_dist}")
    print(f"  [regime] Test : {test_regime_dist}")

    # State distribution from HMM
    dist = hmm.state_distribution(X_train_reg)
    if config.get("hmm", {}).get("n_states", 3) == 3:
        if dist.get("sideways", 0) < 0.10:
            print("[pipeline] WARNING: sideways < 10% of bars")

    # ---- 5. Build signal dataframes ----
    if mode == "multi_tf":
        # Forward-fill regime TF labels onto signal TF bars.
        print(f"[pipeline] Forward-filling {regime_tf} regime onto {signal_tf} bars...")
        train_sig_df = forward_fill_1h_regime(train_sig_feat, train_labels, train_bull, train_bear, idx_train_reg,
                                               atr_1h=train_reg_feat["atr_14"].rename(None),
                                               hma_rising=train_reg_feat["hma_rising"].rename(None),
                                               price_above_hma=train_reg_feat["price_above_hma"].rename(None),
                                               hmm_sideways_prob=train_sideways)
        val_sig_df   = forward_fill_1h_regime(val_sig_feat,   val_labels,   val_bull,   val_bear,   idx_val_reg,
                                               atr_1h=val_reg_feat["atr_14"].rename(None),
                                               hma_rising=val_reg_feat["hma_rising"].rename(None),
                                               price_above_hma=val_reg_feat["price_above_hma"].rename(None),
                                               hmm_sideways_prob=val_sideways)
        test_sig_df  = forward_fill_1h_regime(test_sig_feat,  test_labels,  test_bull,  test_bear,  idx_test_reg,
                                               atr_1h=test_reg_feat["atr_14"].rename(None),
                                               hma_rising=test_reg_feat["hma_rising"].rename(None),
                                               price_above_hma=test_reg_feat["price_above_hma"].rename(None),
                                               hmm_sideways_prob=test_sideways)
        freq = _TF_TO_FREQ.get(signal_tf, "5min")
    else:
        train_sig_df = train_reg_feat
        val_sig_df   = val_reg_feat
        test_sig_df  = test_reg_feat
        freq         = "1h"

    # Generate signals
    print("[pipeline] Generating signals...")
    strategy_mode = "legacy" if legacy_signals else config.get("strategies", {}).get("mode", "legacy")
    sig_kwargs = dict(
        config                  = config,
        adx_threshold           = p["adx_threshold"],
        hmm_min_prob            = p["hmm_min_prob"],
        regime_persistence_bars = p["regime_persistence_bars"],
        require_smc_confluence  = p["require_smc_confluence"],
        require_pin_bar         = p["require_pin_bar"],
    )

    if strategy_mode == "regime_specialized":
        print(f"  [pipeline] strategy_mode=regime_specialized")
        if mode == "multi_tf":
            train_sig = route_signals(train_sig_df, config, hmm_model=hmm)
            val_sig   = route_signals(val_sig_df,   config, hmm_model=hmm)
            test_sig  = route_signals(test_sig_df,  config, hmm_model=hmm)
        else:
            train_sig = route_signals(train_sig_df, config,
                hmm_labels=train_labels, hmm_bull_prob=train_bull, hmm_bear_prob=train_bear, hmm_index=idx_train_reg, hmm_model=hmm)
            val_sig   = route_signals(val_sig_df, config,
                hmm_labels=val_labels,   hmm_bull_prob=val_bull,   hmm_bear_prob=val_bear,   hmm_index=idx_val_reg, hmm_model=hmm)
            test_sig  = route_signals(test_sig_df, config,
                hmm_labels=test_labels,  hmm_bull_prob=test_bull,  hmm_bear_prob=test_bear,  hmm_index=idx_test_reg, hmm_model=hmm)
        train_sig = compute_stops_regime_aware(train_sig, config)
        val_sig   = compute_stops_regime_aware(val_sig,   config)
        test_sig  = compute_stops_regime_aware(test_sig,  config)
    else:
        print(f"  [pipeline] strategy_mode=legacy")
        if mode == "multi_tf":
            train_sig = generate_signals(train_sig_df, **sig_kwargs)
            val_sig   = generate_signals(val_sig_df,   **sig_kwargs)
            test_sig  = generate_signals(test_sig_df,  **sig_kwargs)
        else:
            train_sig = generate_signals(
                train_sig_df, **sig_kwargs,
                hmm_labels=train_labels, hmm_bull_prob=train_bull, hmm_bear_prob=train_bear, hmm_index=idx_train_reg,
            )
            val_sig   = generate_signals(
                val_sig_df, **sig_kwargs,
                hmm_labels=val_labels,   hmm_bull_prob=val_bull,   hmm_bear_prob=val_bear,   hmm_index=idx_val_reg,
            )
            test_sig  = generate_signals(
                test_sig_df, **sig_kwargs,
                hmm_labels=test_labels,  hmm_bull_prob=test_bull,  hmm_bear_prob=test_bear,  hmm_index=idx_test_reg,
            )
        train_sig = compute_stops(train_sig, p["atr_stop_mult"], p["atr_tp_mult"])
        val_sig   = compute_stops(val_sig,   p["atr_stop_mult"], p["atr_tp_mult"])
        test_sig  = compute_stops(test_sig,  p["atr_stop_mult"], p["atr_tp_mult"])

    print(f"  Train: long={int(train_sig['signal_long'].sum())} | short={int(train_sig['signal_short'].sum())}")
    print(f"  Test : long={int(test_sig['signal_long'].sum())}  | short={int(test_sig['signal_short'].sum())}")

    # Diagnostic: probabilistic gating stats on test split
    _log_signal_stats(test_sig, config)

    # ---- 5b. Optional: Optuna optimization on val ----
    if optimize:
        print(f"\n[pipeline] Running Optuna optimization ({n_trials} trials) targeting {optuna_target}...")
        # When legacy_signals=True, optimizer must also use legacy mode (consistent objective)
        import copy as _copy
        opt_config = config
        if legacy_signals and config.get("strategies", {}).get("mode") != "legacy":
            opt_config = _copy.deepcopy(config)
            opt_config.setdefault("strategies", {})["mode"] = "legacy"
        best_params, best_val_sharpe = run_optimization(
            val_sig_df   = val_sig_df,
            config       = opt_config,
            n_trials     = n_trials,
            train_sig_df = train_sig_df,  # combined train+val objective avoids overfitting
            optuna_target = optuna_target,
        )

        # Override p with best params
        if best_val_sharpe > -990:
            p.update(best_params)
            print(f"[pipeline] Applying best params (val_sharpe={best_val_sharpe:.4f})")

            # Regenerate signals + stops with best params
            sig_kwargs_opt = dict(
                config                  = config,
                adx_threshold           = p.get("adx_threshold",           p["adx_threshold"]),
                hmm_min_prob            = p.get("hmm_min_prob",             p["hmm_min_prob"]),
                regime_persistence_bars = p.get("regime_persistence_bars", p["regime_persistence_bars"]),
                require_smc_confluence  = p.get("require_smc_confluence",  p["require_smc_confluence"]),
                require_pin_bar         = p.get("require_pin_bar",         p["require_pin_bar"]),
            )
            if strategy_mode == "regime_specialized":
                if mode == "multi_tf":
                    train_sig = route_signals(train_sig_df, config, hmm_model=hmm)
                    val_sig   = route_signals(val_sig_df,   config, hmm_model=hmm)
                    test_sig  = route_signals(test_sig_df,  config, hmm_model=hmm)
                else:
                    train_sig = route_signals(train_sig_df, config,
                        hmm_labels=train_labels, hmm_bull_prob=train_bull, hmm_bear_prob=train_bear, hmm_index=idx_train_reg, hmm_model=hmm)
                    val_sig   = route_signals(val_sig_df, config,
                        hmm_labels=val_labels,   hmm_bull_prob=val_bull,   hmm_bear_prob=val_bear,   hmm_index=idx_val_reg, hmm_model=hmm)
                    test_sig  = route_signals(test_sig_df, config,
                        hmm_labels=test_labels,  hmm_bull_prob=test_bull,  hmm_bear_prob=test_bear,  hmm_index=idx_test_reg, hmm_model=hmm)
                train_sig = compute_stops_regime_aware(train_sig, config)
                val_sig   = compute_stops_regime_aware(val_sig,   config)
                test_sig  = compute_stops_regime_aware(test_sig,  config)
            else:
                if mode == "multi_tf":
                    train_sig = generate_signals(train_sig_df, **sig_kwargs_opt)
                    val_sig   = generate_signals(val_sig_df,   **sig_kwargs_opt)
                    test_sig  = generate_signals(test_sig_df,  **sig_kwargs_opt)
                else:
                    train_sig = generate_signals(train_sig_df, **sig_kwargs_opt,
                        hmm_labels=train_labels, hmm_bull_prob=train_bull, hmm_bear_prob=train_bear, hmm_index=idx_train_reg)
                    val_sig   = generate_signals(val_sig_df,   **sig_kwargs_opt,
                        hmm_labels=val_labels,   hmm_bull_prob=val_bull,   hmm_bear_prob=val_bear,   hmm_index=idx_val_reg)
                    test_sig  = generate_signals(test_sig_df,  **sig_kwargs_opt,
                        hmm_labels=test_labels,  hmm_bull_prob=test_bull,  hmm_bear_prob=test_bear,  hmm_index=idx_test_reg)
                train_sig = compute_stops(train_sig, p["atr_stop_mult"], p["atr_tp_mult"])
                val_sig   = compute_stops(val_sig,   p["atr_stop_mult"], p["atr_tp_mult"])
                test_sig  = compute_stops(test_sig,  p["atr_stop_mult"], p["atr_tp_mult"])
            print(f"  Optimized: long={int(test_sig['signal_long'].sum())} | short={int(test_sig['signal_short'].sum())} (test)")

    # ---- 6. Backtests ----
    print("\n[pipeline] Running TRAIN backtest...")
    train_metrics, _ = run_backtest(train_sig, strategy_name, "train", config, dirs["backtests"], freq=freq)

    print("\n[pipeline] Running VAL backtest...")
    val_metrics,   _ = run_backtest(val_sig,   strategy_name, "val",   config, dirs["backtests"], freq=freq)

    print("\n[pipeline] Running TEST backtest (out-of-sample)...")
    test_metrics, test_trades_df = run_backtest(test_sig, strategy_name, "test", config, dirs["backtests"], freq=freq)

    print("\n[pipeline] Running 2x cost sensitivity test...")
    test_2x_metrics = run_backtest_2x_costs(test_sig, strategy_name, "test", config, dirs["backtests"], freq=freq)
    test_metrics["cost_sensitivity_2x"] = {
        "sharpe_ratio":      test_2x_metrics.get("sharpe_ratio"),
        "annualized_return": test_2x_metrics.get("annualized_return"),
        "max_drawdown":      test_2x_metrics.get("max_drawdown"),
    }

    # ---- 7. Walk-forward ----
    wf_results = None
    if walk_forward:
        print("\n[pipeline] Running walk-forward validation...")
        wf_results = run_walk_forward(
            strategy_name, config, raw_regime_path, dirs["backtests"],
            freq=_TF_TO_FREQ.get(regime_tf, "1h"),
        )
        if wf_results.get("available"):
            print(f"[pipeline] Walk-forward: mean_sharpe={wf_results.get('mean_sharpe',0):.3f} | positive={wf_results.get('pct_positive',0)*100:.0f}%")

    # ---- 8. Hard rejection checks ----
    hrd = hard_rejection_checks(
        metrics                  = test_metrics,
        config                   = config,
        train_regime_dist        = train_regime_dist,
        test_regime_dist         = test_regime_dist,
        cost_sensitivity_metrics = test_2x_metrics,
        walk_forward_run         = (wf_results is not None and wf_results.get("available", False)),
        train_metrics            = train_metrics,
        val_metrics              = val_metrics,
    )
    if hrd["hard_rejected"]:
        print(f"\n[pipeline] HARD REJECTION triggered:")
        for k, v in hrd["rejections"].items():
            print(f"  - {k}: {v}")

    # ---- 9. Multi-split verdict ----
    mv            = multi_split_verdict(train_metrics, val_metrics, test_metrics, config, wf_results, hrd)
    final_verdict = mv["verdict"]

    for flag, active in mv["flags"].items():
        if active:
            print(f"[pipeline] FLAG: {flag.replace('_',' ').upper()}")

    # ---- 10. Experiment logging ----
    exp = ExperimentLogger(dirs["experiments"])
    exp.log(
        params        = p,
        config        = config,
        train_metrics = train_metrics,
        val_metrics   = val_metrics,
        test_metrics  = test_metrics,
        verdict       = final_verdict,
        wf_results    = wf_results,
        hard_checks   = hrd,
        extra         = {"gap_audit": gap_audit, "dataset": data_ver},
    )

    results = {
        "strategy_name":  strategy_name,
        "verdict":        final_verdict,
        "verdict_detail": mv,
        "date":           str(date.today()),
        "experiment_id":  exp.run_id,
        "params":         p,
        "dataset":        data_ver,
        "gap_audit":      gap_audit,
        "regime_distributions": {"train": train_regime_dist, "test": test_regime_dist},
        "train_metrics":  train_metrics,
        "val_metrics":    val_metrics,
        "test_metrics":   test_metrics,
        "walk_forward":   wf_results,
    }

    if return_trades and test_trades_df is not None and len(test_trades_df) > 0:
        results["test_trades"] = test_trades_df.to_dict(orient="records")

    # Save final verdict JSON
    dirs["reports"].mkdir(parents=True, exist_ok=True)
    verdict_path = dirs["reports"] / "final_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[pipeline] Final verdict saved to {verdict_path}")

    # ---- 11. Export if approved + flag set ----
    if export_approved and final_verdict == "APPROVED":
        export_dir = export_approved_strategy(
            results       = results,
            config        = config,
            artefacts_dir = dirs["root"],
            model_path    = model_path,
            strategy_name = strategy_name,
        )
        results["exported_to"] = str(export_dir)
        print(f"\n[pipeline] Strategy APPROVED and EXPORTED to {export_dir}")

    _print_summary(train_metrics, val_metrics, test_metrics, final_verdict, mv)
    return results


def _print_summary(train_m, val_m, test_m, v, mv):
    SEP  = "=" * 65
    DASH = "-" * 65

    def _r(m, k, pct=False):
        val = m.get(k, 0)
        return f"{val*100:>8.2f}%" if pct else f"{val:>8.3f}"

    print(f"\n{SEP}")
    print(f"  XAUUSD STRATEGY  |  MULTI-SPLIT RESULTS")
    print(f"{SEP}")
    print(f"  {'Metric':<26} {'TRAIN':>10} {'VAL':>10} {'TEST':>10}")
    print(f"{DASH}")
    print(f"  {'Annualized Return':<26} {_r(train_m,'annualized_return',True):>10} {_r(val_m,'annualized_return',True):>10} {_r(test_m,'annualized_return',True):>10}")
    print(f"  {'Sharpe Ratio':<26} {_r(train_m,'sharpe_ratio'):>10} {_r(val_m,'sharpe_ratio'):>10} {_r(test_m,'sharpe_ratio'):>10}")
    print(f"  {'Max Drawdown':<26} {_r(train_m,'max_drawdown',True):>10} {_r(val_m,'max_drawdown',True):>10} {_r(test_m,'max_drawdown',True):>10}")
    print(f"  {'Profit Factor':<26} {_r(train_m,'profit_factor'):>10} {_r(val_m,'profit_factor'):>10} {_r(test_m,'profit_factor'):>10}")
    print(f"  {'Total Trades':<26} {train_m.get('total_trades','N/A'):>10} {val_m.get('total_trades','N/A'):>10} {test_m.get('total_trades','N/A'):>10}")
    print(f"  {'Win Rate':<26} {_r(train_m,'win_rate',True):>10} {_r(val_m,'win_rate',True):>10} {_r(test_m,'win_rate',True):>10}")
    cs = test_m.get("cost_sensitivity_2x", {})
    if cs:
        print(f"  {'Sharpe @ 2x costs':<26} {'':>10} {'':>10} {cs.get('sharpe_ratio', 0):>10.3f}")
    print(f"{DASH}")
    print(f"  Split OK:  train={mv['train_ok']}  val={mv['val_ok']}  test={mv['test_ok']}  wf={mv['wf_ok']}")
    hrd = mv.get("hard_checks", {})
    if hrd.get("hard_rejected"):
        for reason in hrd.get("rejections", {}).values():
            print(f"  ! HARD REJECT: {reason}")
    for flag, active in mv["flags"].items():
        if active:
            print(f"  ! FLAG: {flag.replace('_',' ').upper()}")
    print(f"{SEP}")
    print(f"  VERDICT: {v}")
    print(f"{SEP}\n")


def main():
    parser = argparse.ArgumentParser(description="XAUUSD Strategy Pipeline (trade2)")
    parser.add_argument("--config",             default="configs/xauusd_mtf.yaml", help="Override config path")
    parser.add_argument("--base-config",        default="configs/base.yaml",       help="Base config path")
    parser.add_argument("--retrain-model",      action="store_true",               help="Force HMM retrain")
    parser.add_argument("--skip-walk-forward",  action="store_true",               help="Skip walk-forward validation")
    parser.add_argument("--optimize",           action="store_true",               help="Run Optuna optimization first")
    parser.add_argument("--trials",             type=int, default=100,             help="Optuna trial count")
    parser.add_argument("--export-approved",    action="store_true",               help="Export strategy if APPROVED")
    parser.add_argument("--legacy-signals",     action="store_true",               help="Force legacy signal generation (ignore strategies.mode)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    base_path     = PROJECT_ROOT / args.base_config
    override_path = PROJECT_ROOT / args.config if args.config != args.base_config else None

    config = load_config(base_path, override_path)

    results = run_pipeline(
        config          = config,
        walk_forward    = not args.skip_walk_forward,
        retrain_model   = args.retrain_model,
        export_approved = args.export_approved,
        optimize        = args.optimize,
        n_trials        = args.trials,
        legacy_signals  = args.legacy_signals,
    )
    sys.exit(0 if results["verdict"] == "APPROVED" else 1)


if __name__ == "__main__":
    main()
