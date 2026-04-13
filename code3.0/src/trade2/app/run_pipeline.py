"""
app/run_pipeline.py - CLI entry point and thin orchestrator.
Replaces both src/pipeline.py and src_v2/pipeline.py.

Usage:
    trade2 --config configs/xauusd_mtf.yaml --retrain-model
    trade2 --skip-walk-forward
    trade2 --optimize --trials 200
    trade2 --retrain-model --export-approved
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path

from trade2.config.loader import load_config
from trade2.data.splits import load_split_tf
from trade2.data.loader import dataset_version
from trade2.data.validation import audit_missing_bars
from trade2.features.builder import add_1h_features, add_5m_features, add_5m_full_features
from trade2.features.hmm_features import get_hmm_feature_matrix
from trade2.models.hmm import XAUUSDRegimeModel
from trade2.signals.regime import forward_fill_1h_regime, forward_fill_5m_to_1m
from trade2.backtesting import pullback_engine
from trade2.signals.generator import generate_signals, compute_stops, compute_stops_regime_aware
from trade2.signals.router import route_signals, apply_tv_signal_filter, ffill_tv_cols_to_5m
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


def _backup_model(model_path: Path, models_dir: Path) -> None:
    """Backup existing model to models/backups/ before retraining."""
    backup_dir = models_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{model_path.stem}_{ts}.pkl"
    shutil.copy(model_path, backup_path)
    print(f"[pipeline] Backed up existing model -> {backup_path}")


def _save_golden_model(model_path: Path, models_dir: Path, test_metrics: dict, strategy_name: str) -> None:
    """Copy model to models/golden/ with a metrics sidecar when return threshold is met."""
    golden_dir = models_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    today  = date.today().strftime("%Y_%m_%d")
    ret    = test_metrics.get("annualized_return", 0) * 100
    sharpe = test_metrics.get("sharpe_ratio", 0)
    name   = f"{model_path.stem}_{today}_ret{ret:.0f}pct_sh{sharpe:.2f}"
    golden_path = golden_dir / f"{name}.pkl"
    shutil.copy(model_path, golden_path)
    sidecar = {
        "strategy":     strategy_name,
        "date":         today,
        "source_model": str(model_path),
        "test_metrics": test_metrics,
    }
    (golden_dir / f"{name}_metrics.json").write_text(json.dumps(sidecar, indent=2, default=str))
    print(f"[pipeline] Golden model saved -> {golden_path}")


def _run_3tf_split(
    df_1m: "pd.DataFrame",
    df_5m_signals: "pd.DataFrame",
    config: dict,
    period_label: str,
    strategy_name: str,
    backtests_dir: Path,
) -> dict:
    """
    Run one train/val/test split of the 3-TF pullback backtest.

    Args:
        df_1m:          1M OHLCV split (date-sliced to match df_5m_signals period)
        df_5m_signals:  5M df after generate_signals + compute_stops (same period)
        config:         full config dict
        period_label:   'train', 'val', or 'test'
        strategy_name:  used for output file naming
        backtests_dir:  where to save CSV / JSON

    Returns:
        metrics dict
    """
    import pandas as pd
    from trade2.backtesting.metrics import compute_metrics
    from trade2.backtesting.costs import compute_slippage_array

    risk_cfg = config["risk"]
    costs    = config["costs"]
    pb_cfg   = config["pullback"]

    df_1m_ready = forward_fill_5m_to_1m(df_1m, df_5m_signals)

    slippage_arr = compute_slippage_array(df_1m_ready["Close"], config).values

    commission_rt = costs["commission_rt_bps"] / 10000.0 * 2

    init_cash = risk_cfg["init_cash"]

    equity, trades_df = pullback_engine.simulate(
        df_1m_ready, pb_cfg, risk_cfg, init_cash, commission_rt, slippage_arr,
    )

    # 1M bars: 252 trading days * 390 minutes/day
    metrics = compute_metrics(equity, trades_df, bars_per_year=98280)

    if backtests_dir is not None:
        trades_path = backtests_dir / f"{strategy_name}_{period_label}_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        metrics_path = backtests_dir / f"{strategy_name}_{period_label}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    ret  = metrics.get("annualized_return", 0) * 100
    sh   = metrics.get("sharpe_ratio", 0)
    dd   = metrics.get("max_drawdown", 0) * 100
    n    = metrics.get("total_trades", 0)
    wr   = metrics.get("win_rate", 0) * 100
    print(f"  [{period_label}] return={ret:.1f}% sharpe={sh:.2f} dd={dd:.1f}% trades={n} wr={wr:.1f}%")
    return metrics


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
    model_path_override: str = None,
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
    _TF_TO_FREQ = {"1M": "1min", "5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"}

    print(f"[pipeline] Loading regime TF ({regime_tf}) data...")
    train_reg, val_reg, test_reg = load_split_tf(regime_tf, config)

    raw_1h_path = DATA_ROOT / config.get("data", {}).get("raw_1h_csv", "data/raw/XAUUSD_1H_2019_2025.csv")

    # Resolve raw CSV for regime TF (needed for walk-forward which reloads its own data)
    _TF_TO_RAW_KEY = {
        "1H":  "raw_1h_csv",
        "5M":  "raw_5m_csv",
        "4H":  "raw_4h_csv",
        "1M":  "raw_1m_csv",
        "15M": "raw_15m_csv",
    }
    _TF_DEFAULTS = {
        "1H":  "data/raw/XAUUSD_1H_2019_2025.csv",
        "5M":  "data/raw/XAUUSD_5M_2019_2025.csv",
        "4H":  "data/raw/XAUUSD_4H_2019_2025.csv",
        "1M":  "code3.0/data/raw/XAUUSD_1M_2019_2026.csv",
        "15M": "data/raw/XAUUSD_15M_2019_2026.csv",
    }
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
    # For single-TF mode with sub-hourly regime TF (5M, 15M), use add_5m_full_features
    # which combines 5M signal features (SMC, SD mean, etc.) with HMM input features.
    # For 1H or 4H (or multi-TF where regime_tf != signal_tf), use add_1h_features.
    _SUB_HOUR_TFS = {"5M", "15M", "1M", "30M"}
    _regime_feat_fn = add_5m_full_features if (mode == "single_tf" and regime_tf in _SUB_HOUR_TFS) else add_1h_features
    _regime_feat_label = "5M signal+HMM" if _regime_feat_fn is add_5m_full_features else regime_tf
    print(f"[pipeline] Engineering {_regime_feat_label} regime features...")
    train_reg_feat = _load_features_cached(train_reg, regime_tf, "train", config, dirs, _regime_feat_fn)
    val_reg_feat   = _load_features_cached(val_reg,   regime_tf, "val",   config, dirs, _regime_feat_fn)
    test_reg_feat  = _load_features_cached(test_reg,  regime_tf, "test",  config, dirs, _regime_feat_fn)

    if mode == "multi_tf":
        smc_label = "" if not config.get("smc_5m", {}).get("enabled", True) else " (SMC)"
        print(f"[pipeline] Engineering {signal_tf} signal features{smc_label}...")
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
    _model_id     = config.get("hmm", {}).get("model_id", "")
    _model_suffix = f"_{_model_id}" if _model_id else ""
    model_path = dirs["models"] / f"hmm_{_model_tf_key}_{p['hmm_states']}states{_model_suffix}.pkl"

    if model_path_override:
        override_p = Path(model_path_override)
        if not override_p.exists():
            raise FileNotFoundError(f"[pipeline] --model-path not found: {override_p}")
        print(f"[pipeline] Loading model from override path: {override_p}")
        hmm = XAUUSDRegimeModel.load(override_p)
    elif retrain_model or not model_path.exists():
        # Backup existing model before overwriting so golden strategies are never lost
        if model_path.exists() and retrain_model:
            _backup_model(model_path, dirs["models"])
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
        # Forward-fill TV indicator _bull/_bear columns from 1H regime into 5M signal DFs
        train_sig_df = ffill_tv_cols_to_5m(train_sig_df, train_reg_feat)
        val_sig_df   = ffill_tv_cols_to_5m(val_sig_df,   val_reg_feat)
        test_sig_df  = ffill_tv_cols_to_5m(test_sig_df,  test_reg_feat)
        # Forward-fill CDC levels from regime TF + compute retest features (cdc_retest strategy)
        if config.get("strategies", {}).get("cdc_retest", {}).get("enabled", False):
            from trade2.signals.regime import forward_fill_cdc_levels
            from trade2.features.cdc_retest import add_cdc_retest_features
            print(f"[pipeline] Forward-filling CDC levels ({regime_tf} -> {signal_tf}) for cdc_retest...")
            train_sig_df = forward_fill_cdc_levels(train_sig_df, train_reg_feat)
            val_sig_df   = forward_fill_cdc_levels(val_sig_df,   val_reg_feat)
            test_sig_df  = forward_fill_cdc_levels(test_sig_df,  test_reg_feat)
            train_sig_df = add_cdc_retest_features(train_sig_df, config)
            val_sig_df   = add_cdc_retest_features(val_sig_df,   config)
            test_sig_df  = add_cdc_retest_features(test_sig_df,  config)
        freq = _TF_TO_FREQ.get(signal_tf, "5min")
    else:
        train_sig_df = train_reg_feat
        val_sig_df   = val_reg_feat
        test_sig_df  = test_reg_feat
        freq         = "1h"

    # ---- 5a. Train / load XGBoost reversal model (optional) ----
    _xgb_cfg = config.get("strategies", {}).get("smc_sd_mean", {}).get("reversal_xgb", {})
    if _xgb_cfg.get("enabled", False):
        from trade2.models.reversal_xgb import ReversalXGBModel
        _xgb_model_path = dirs["models"] / "reversal_xgb_sd_mean.pkl"
        if retrain_model or not _xgb_model_path.exists():
            print("[pipeline] Training XGBoost reversal detector on train split...")
            _xgb_model = ReversalXGBModel(config)
            _xgb_model.fit(train_sig_df if mode == "multi_tf" else train_sig_df)
            _xgb_model.save(_xgb_model_path)
        else:
            print(f"[pipeline] Loading XGBoost reversal model from {_xgb_model_path}")
            _xgb_model = ReversalXGBModel.load(_xgb_model_path)

        # Attach reversal_prob columns to all three signal DataFrames
        def _attach_reversal_probs(df):
            df = df.copy()
            df["reversal_prob_long"]  = _xgb_model.predict_proba_long(df)
            df["reversal_prob_short"] = _xgb_model.predict_proba_short(df)
            return df

        if mode == "multi_tf":
            train_sig_df = _attach_reversal_probs(train_sig_df)
            val_sig_df   = _attach_reversal_probs(val_sig_df)
            test_sig_df  = _attach_reversal_probs(test_sig_df)
        else:
            train_sig_df = _attach_reversal_probs(train_sig_df)
            val_sig_df   = _attach_reversal_probs(val_sig_df)
            test_sig_df  = _attach_reversal_probs(test_sig_df)

        # Log prob distribution on test split
        _tpl = test_sig_df["reversal_prob_long"].dropna()
        _tps = test_sig_df["reversal_prob_short"].dropna()
        print(f"  [reversal_xgb] test long  prob: mean={_tpl.mean():.3f} p25={_tpl.quantile(0.25):.3f} p75={_tpl.quantile(0.75):.3f}")
        print(f"  [reversal_xgb] test short prob: mean={_tps.mean():.3f} p25={_tps.quantile(0.25):.3f} p75={_tps.quantile(0.75):.3f}")

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
        train_sig = apply_tv_signal_filter(train_sig, config)
        val_sig   = apply_tv_signal_filter(val_sig,   config)
        test_sig  = apply_tv_signal_filter(test_sig,  config)
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
        _use_sig_atr = config["risk"]["use_signal_atr"]
        train_sig = compute_stops(train_sig, p["atr_stop_mult"], p["atr_tp_mult"], use_signal_atr=_use_sig_atr)
        val_sig   = compute_stops(val_sig,   p["atr_stop_mult"], p["atr_tp_mult"], use_signal_atr=_use_sig_atr)
        test_sig  = compute_stops(test_sig,  p["atr_stop_mult"], p["atr_tp_mult"], use_signal_atr=_use_sig_atr)

    print(f"  Train: long={int(train_sig['signal_long'].sum())} | short={int(train_sig['signal_short'].sum())}")
    print(f"  Test : long={int(test_sig['signal_long'].sum())}  | short={int(test_sig['signal_short'].sum())}")

    # Diagnostic: probabilistic gating stats on test split
    _log_signal_stats(test_sig, config)

    # ---- 5b. Optional: Optuna optimization on val ----
    if optimize:
        print(f"\n[pipeline] Running Optuna optimization ({n_trials} trials) targeting {optuna_target}...")
        import copy as _copy

        # smc_sd_mean: use dedicated optimizer (SD params require feature recomputation)
        _smc_sd_mean_active = config.get("strategies", {}).get("smc_sd_mean", {}).get("enabled", False)
        if _smc_sd_mean_active:
            from trade2.optimization.optimizer import run_optimization_smc_sd_mean
            # In single_tf mode the feature dfs don't have regime/bull_prob/bear_prob columns
            # yet -- they get passed as separate kwargs in route_signals(). The optimizer calls
            # route_signals() without those kwargs, so we merge them in here.
            if mode == "multi_tf":
                _val_feat   = val_sig_df
                _train_feat = train_sig_df
            else:
                _val_feat   = val_reg_feat.copy()
                _train_feat = train_reg_feat.copy()
                # Merge HMM probabilities so the optimizer's _run_split can call route_signals()
                _val_feat["regime"]    = pd.Series(val_labels,   index=idx_val_reg).reindex(_val_feat.index).fillna("sideways")
                _val_feat["bull_prob"] = pd.Series(val_bull,     index=idx_val_reg).reindex(_val_feat.index).fillna(0.0)
                _val_feat["bear_prob"] = pd.Series(val_bear,     index=idx_val_reg).reindex(_val_feat.index).fillna(0.0)
                _train_feat["regime"]    = pd.Series(train_labels, index=idx_train_reg).reindex(_train_feat.index).fillna("sideways")
                _train_feat["bull_prob"] = pd.Series(train_bull,   index=idx_train_reg).reindex(_train_feat.index).fillna(0.0)
                _train_feat["bear_prob"] = pd.Series(train_bear,   index=idx_train_reg).reindex(_train_feat.index).fillna(0.0)
            best_result = run_optimization_smc_sd_mean(
                val_feat_df   = _val_feat,
                train_feat_df = _train_feat,
                config        = config,
                n_trials      = n_trials,
                optuna_target = optuna_target,
            )
            best_params    = best_result["best_params"]
            best_val_sharpe = best_result["best_value"]
        else:
            # Default optimizer for all other strategies
            opt_config = config
            if legacy_signals and config.get("strategies", {}).get("mode") != "legacy":
                opt_config = _copy.deepcopy(config)
                opt_config.setdefault("strategies", {})["mode"] = "legacy"
            best_params, best_val_sharpe = run_optimization(
                val_sig_df    = val_sig_df,
                config        = opt_config,
                n_trials      = n_trials,
                train_sig_df  = train_sig_df,
                optuna_target = optuna_target,
            )

        # Override p with best params
        if best_val_sharpe > -990:
            p.update(best_params)
            print(f"[pipeline] Applying best params (val_sharpe={best_val_sharpe:.4f})")

            # For smc_sd_mean: patch config["strategies"]["smc_sd_mean"] with best params
            # (optimizer returns flat params; route_signals reads from nested config)
            if _smc_sd_mean_active:
                _smc_cfg = config.get("strategies", {}).get("smc_sd_mean", {})
                _smc_param_keys = [
                    "sd_length_short", "sd_length_long", "sd_atr_threshold",
                    "sd_smooth_length", "sd_smooth_sd_length", "sd_entry_threshold",
                    "atr_stop_mult", "atr_tp_mult", "min_prob", "min_prob_short",
                    "wick_ratio", "cooldown_bars", "bb_zone_threshold", "rsi_upper",
                ]
                for _k in _smc_param_keys:
                    if _k in best_params:
                        _smc_cfg[_k] = best_params[_k]
                if "hmm_min_prob" in best_params:
                    _smc_cfg["min_prob"] = best_params["hmm_min_prob"]
                    _smc_cfg["min_prob_short"] = best_params["hmm_min_prob"]

            # Regenerate signals + stops with best params
            sig_kwargs_opt = dict(
                config                  = config,
                adx_threshold           = p.get("adx_threshold",           p["adx_threshold"]),
                hmm_min_prob            = p.get("hmm_min_prob",             p["hmm_min_prob"]),
                regime_persistence_bars = p.get("regime_persistence_bars", p["regime_persistence_bars"]),
                require_smc_confluence  = p.get("require_smc_confluence",  p["require_smc_confluence"]),
                require_pin_bar         = p.get("require_pin_bar",         p["require_pin_bar"]),
            )
            # Re-compute SD features if smc_sd_mean is active (SD params may have changed)
            if _smc_sd_mean_active and any(k in best_params for k in [
                "sd_length_short", "sd_length_long", "sd_atr_threshold",
                "sd_smooth_length", "sd_smooth_sd_length",
            ]):
                from trade2.features.sd_adaptive_mean import compute_sd_adaptive_mean
                print("[pipeline] Re-computing SD Adaptive Mean features with optimized params...")
                train_sig_df = compute_sd_adaptive_mean(train_sig_df, config)
                val_sig_df   = compute_sd_adaptive_mean(val_sig_df,   config)
                test_sig_df  = compute_sd_adaptive_mean(test_sig_df,  config)

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
                train_sig = apply_tv_signal_filter(train_sig, config)
                val_sig   = apply_tv_signal_filter(val_sig,   config)
                test_sig  = apply_tv_signal_filter(test_sig,  config)
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
                _use_sig_atr = config["risk"]["use_signal_atr"]
                train_sig = compute_stops(train_sig, p["atr_stop_mult"], p["atr_tp_mult"], use_signal_atr=_use_sig_atr)
                val_sig   = compute_stops(val_sig,   p["atr_stop_mult"], p["atr_tp_mult"], use_signal_atr=_use_sig_atr)
                test_sig  = compute_stops(test_sig,  p["atr_stop_mult"], p["atr_tp_mult"], use_signal_atr=_use_sig_atr)
            print(f"  Optimized: long={int(test_sig['signal_long'].sum())} | short={int(test_sig['signal_short'].sum())} (test)")

    # ---- 3-TF Pullback mode ----
    entry_tf = config.get("strategy", {}).get("entry_timeframe", None)
    if entry_tf == "1M" and mode == "multi_tf":
        print(f"\n[pipeline] 3-TF mode: loading 1M entry bars...")
        train_1m, val_1m, test_1m = load_split_tf("1M", config)
        print(f"  1M train={len(train_1m)} | val={len(val_1m)} | test={len(test_1m)}")

        print(f"[pipeline] Running 3-TF pullback backtests...")
        train_metrics = _run_3tf_split(train_1m, train_sig, config, "train", strategy_name, dirs["backtests"])
        val_metrics   = _run_3tf_split(val_1m,   val_sig,   config, "val",   strategy_name, dirs["backtests"])
        test_metrics  = _run_3tf_split(test_1m,  test_sig,  config, "test",  strategy_name, dirs["backtests"])

        result = {
            "train": train_metrics,
            "val":   val_metrics,
            "test":  test_metrics,
            "strategy_name": strategy_name,
        }
        print(f"\n[pipeline] 3-TF run complete.")
        return result

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
        if mode == "multi_tf":
            _sig_raw_key = _TF_TO_RAW_KEY.get(signal_tf, "raw_5m_csv")
            _sig_default = _TF_DEFAULTS.get(signal_tf, "data/raw/XAUUSD_5M_2019_2025.csv")
            raw_signal_path = DATA_ROOT / config["data"].get(_sig_raw_key, _sig_default)
        else:
            raw_signal_path = None
        # For single-TF on 5M: use 5M freq. For multi-TF: use signal_tf freq. Else 1h.
        if mode == "multi_tf":
            _wf_freq = _TF_TO_FREQ.get(signal_tf, "5min")
        else:
            _wf_freq = _TF_TO_FREQ.get(regime_tf, "1h")
        wf_results = run_walk_forward(
            strategy_name, config, raw_regime_path, dirs["backtests"],
            freq=_wf_freq,
            raw_signal_path=raw_signal_path,
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

    # ---- 8b. Auto-save golden model if test return meets threshold ----
    golden_threshold = config.get("pipeline", {}).get("golden_model_threshold", 0.20)
    test_return = test_metrics.get("annualized_return", 0)
    if test_return >= golden_threshold and not model_path_override:
        _save_golden_model(model_path, dirs["models"], test_metrics, strategy_name)

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
        _test_csv = dirs["backtests"] / f"{strategy_name}_test_trades.csv"
        export_dir = export_approved_strategy(
            results       = results,
            config        = config,
            artefacts_dir = dirs["root"],
            model_path    = model_path,
            strategy_name = strategy_name,
            trades_csv    = _test_csv,
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
    parser.add_argument("--model-path",         default=None,                      help="Load a specific HMM model .pkl (bypasses default path, skips retrain)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    base_path     = PROJECT_ROOT / args.base_config
    override_path = PROJECT_ROOT / args.config if args.config != args.base_config else None

    config = load_config(base_path, override_path)

    results = run_pipeline(
        config               = config,
        walk_forward         = not args.skip_walk_forward,
        retrain_model        = args.retrain_model,
        export_approved      = args.export_approved,
        optimize             = args.optimize,
        n_trials             = args.trials,
        legacy_signals       = args.legacy_signals,
        model_path_override  = args.model_path,
    )
    sys.exit(0 if results["verdict"] == "APPROVED" else 1)


if __name__ == "__main__":
    main()
