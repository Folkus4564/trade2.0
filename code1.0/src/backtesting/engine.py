"""
Module: engine.py
Purpose: vectorbt-based backtesting engine for XAUUSD systematic strategy
Author: Strategy Code Engineer Agent
Date: 2026-03-08
"""

import json
import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parents[2]))
from src.config import get_config
from src.backtesting.metrics import compute_metrics, format_report, verdict, passes_criteria, compute_random_baseline

# Asian session hours (UTC) — wider spreads
_ASIAN_HOURS = set(range(0, 7)) | set(range(22, 24))

BACKTESTS_DIR = Path(__file__).parents[2] / "backtests"
BACKTESTS_DIR.mkdir(exist_ok=True)


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    period_label: str = "test",
    init_cash: float = 100_000.0,
    size_pct: float  = 0.95,
) -> Tuple[Dict[str, Any], vbt.Portfolio]:
    """
    Run vectorbt backtest using signal columns already in df.

    Expects these columns in df:
    - signal_long, signal_short  (1/0)
    - exit_long, exit_short      (1/0)

    Args:
        df:             Feature+signal DataFrame
        strategy_name:  Name for saving results
        period_label:   'train', 'val', or 'test'
        init_cash:      Starting capital
        size_pct:       Fraction of portfolio per trade

    Returns:
        (metrics_dict, vbt.Portfolio)
    """
    # ── Load costs from config ─────────────────────────────────────────────────
    cfg = get_config()
    costs_cfg = cfg.get("costs", {})
    SPREAD_PIPS    = costs_cfg.get("spread_pips",   3)
    SLIPPAGE_PIPS  = costs_cfg.get("slippage_pips", 1)
    COMMISSION_RT  = costs_cfg.get("commission_rt", 0.0002)
    ASIAN_MULT     = costs_cfg.get("spread_asian_mult", 2.5)
    VOL_MULT       = costs_cfg.get("spread_vol_mult",   1.5)
    ATR_LB         = costs_cfg.get("spread_vol_atr_lookback", 20)
    SPREAD_PIPS_WIDE = SPREAD_PIPS * ASIAN_MULT

    close = df["Close"].astype(float)

    # ── Signals ───────────────────────────────────────────────────────────────
    long_entries  = df["signal_long"].astype(bool)
    long_exits    = df["exit_long"].astype(bool)
    short_entries = df["signal_short"].astype(bool)
    short_exits   = df["exit_short"].astype(bool)

    # Ensure no simultaneous long and short entries
    conflict = long_entries & short_entries
    long_entries  = long_entries  & ~conflict
    short_entries = short_entries & ~conflict

    if long_entries.sum() == 0 and short_entries.sum() == 0:
        print(f"[engine] WARNING: No signals generated for {period_label}")
        empty_metrics = {
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
        return empty_metrics, None

    # ── Dynamic spread expansion (Phase C1) ──────────────────────────────────
    # Per-bar spread: Asian hours * 2.5x, high vol * 1.5x, compound if both
    avg_price = close.mean()
    pip_value = 0.01  # XAUUSD 1 pip = $0.01

    if hasattr(close.index, 'hour'):
        idx_hours = close.index.hour if close.index.tz is None else close.index.tz_convert("UTC").hour
        is_asian  = pd.Series(idx_hours, index=close.index).isin(_ASIAN_HOURS)
    else:
        is_asian  = pd.Series(False, index=close.index)

    # ATR-based volatility filter
    if "atr_14" in df.columns:
        atr     = df["atr_14"].astype(float)
        atr_ma  = atr.rolling(ATR_LB, min_periods=1).mean()
        is_hvol = atr > (1.5 * atr_ma)
    else:
        is_hvol = pd.Series(False, index=close.index)

    # Compound spread per bar
    spread_arr = pd.Series(float(SPREAD_PIPS), index=close.index)
    spread_arr = np.where(is_asian & is_hvol, spread_arr * ASIAN_MULT * VOL_MULT,
                 np.where(is_asian,            spread_arr * ASIAN_MULT,
                 np.where(is_hvol,             spread_arr * VOL_MULT,
                                               spread_arr)))
    spread_arr = pd.Series(spread_arr, index=close.index)

    # Per-bar slippage fraction
    slippage_arr = ((spread_arr + SLIPPAGE_PIPS) * pip_value) / (avg_price + 1e-10)
    # vectorbt accepts a scalar slippage; use blended average for simplicity
    slippage = float(slippage_arr.mean())

    # ── Run portfolio ─────────────────────────────────────────────────────────
    # Use fixed value sizing (init_cash * size_pct per trade) instead of
    # "percent" which does not support position reversal in vectorbt.
    trade_value = init_cash * size_pct

    pf = vbt.Portfolio.from_signals(
        close         = close,
        entries       = long_entries,
        exits         = long_exits,
        short_entries = short_entries,
        short_exits   = short_exits,
        init_cash     = init_cash,
        fees          = COMMISSION_RT / 2,   # per side
        slippage      = slippage,
        size          = trade_value,
        size_type     = "value",
        freq          = "1h",
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    equity = pf.value()
    trades_df = pf.trades.records_readable

    if len(trades_df) > 0:
        trade_records = pd.DataFrame({
            "pnl": trades_df.get("PnL", pd.Series(dtype=float)),
            "duration_bars": trades_df.get("Duration", pd.Series(dtype=float)),
        })
    else:
        trade_records = None

    # Buy-and-hold benchmark
    bh_equity = init_cash * (close / close.iloc[0])

    metrics = compute_metrics(
        equity_curve      = equity,
        trades            = trade_records,
        benchmark_equity  = bh_equity,
        bars_per_year     = 252 * 24,
    )

    # Random baseline comparison (all splits — Phase B3)
    if metrics.get("total_trades", 0) > 0:
        baseline = compute_random_baseline(
            close        = close,
            n_trades     = metrics["total_trades"],
            bars_per_year= 252 * 24,
        )
        metrics["random_baseline"] = baseline
        strategy_sharpe = metrics.get("sharpe_ratio", 0)
        p95_sharpe      = baseline["random_p95_sharpe"]
        beats_baseline  = strategy_sharpe > p95_sharpe
        metrics["beats_random_baseline"] = beats_baseline
        print(f"[engine] Random baseline (p95 Sharpe): {p95_sharpe:.3f} | Strategy: {strategy_sharpe:.3f} | Beats: {beats_baseline}")

    # Print report
    label = f"{strategy_name} [{period_label}]"
    print(format_report(metrics, label))

    # ── Save results ──────────────────────────────────────────────────────────
    result_path = BACKTESTS_DIR / f"{strategy_name}_{period_label}_results.json"
    save_data = {
        "strategy": strategy_name,
        "period":   period_label,
        "start":    str(df.index[0]),
        "end":      str(df.index[-1]),
        "n_bars":   len(df),
        "metrics":  metrics,
        "pass_criteria": passes_criteria(metrics),
        "verdict":  verdict(metrics),
        "costs": {
            "spread_pips":    SPREAD_PIPS,
            "slippage_pips":  SLIPPAGE_PIPS,
            "commission_rt":  COMMISSION_RT,
        }
    }
    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"[engine] Results saved to {result_path}")

    return metrics, pf


def run_backtest_2x_costs(
    df: pd.DataFrame,
    strategy_name: str,
    period_label: str = "test",
    init_cash: float = 100_000.0,
    size_pct: float  = 0.95,
) -> Dict[str, Any]:
    """
    Run backtest with 2x costs to test cost sensitivity (Phase B3).
    Returns metrics dict only (no portfolio object saved).
    """
    cfg = get_config()
    costs_cfg = cfg.get("costs", {})

    # Temporarily double costs via a monkey-patch approach
    import copy
    import yaml
    doubled_cfg = copy.deepcopy(cfg)
    doubled_cfg["costs"]["spread_pips"]   = costs_cfg.get("spread_pips",   3) * 2
    doubled_cfg["costs"]["slippage_pips"] = costs_cfg.get("slippage_pips", 1) * 2
    doubled_cfg["costs"]["commission_rt"] = costs_cfg.get("commission_rt", 0.0002) * 2

    # Inline the backtest logic with doubled costs
    from src.config import _cached_config as _orig
    import src.config as _cfg_mod
    orig = _cfg_mod._cached_config
    _cfg_mod._cached_config = doubled_cfg
    try:
        metrics, _ = run_backtest(df, strategy_name, period_label + "_2xcost", init_cash, size_pct)
    finally:
        _cfg_mod._cached_config = orig

    return metrics


def run_walk_forward(
    strategy_name: str,
    windows: list = None,
) -> Dict[str, Any]:
    """
    Proper walk-forward validation.
    Each window independently: loads data, adds features, RETRAINS HMM, generates signals, backtests.
    This is the only correct walk-forward — reusing a single trained HMM across windows is invalid.

    Args:
        strategy_name: Name for result files
        windows:       List of dicts with train_start/train_end/val_start/val_end

    Returns:
        Summary dict with per-window metrics and aggregate statistics
    """
    import yaml
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).parents[2]))
    from src.data.loader import load_raw, split as _split, RAW_CSV
    from src.data.features import add_features, get_hmm_feature_matrix
    from src.models.hmm_model import XAUUSDRegimeModel
    from src.models.signal_generator import generate_signals, compute_stops

    cfg_path = _Path(__file__).parents[2] / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if windows is None:
        windows = cfg["walk_forward"]["windows"]

    p = cfg.get("hmm", {})
    raw = load_raw(RAW_CSV)

    results = []
    for i, win in enumerate(windows):
        te = pd.Timestamp(win["train_end"], tz="UTC")
        vs = pd.Timestamp(win["val_start"], tz="UTC")
        ve = pd.Timestamp(win["val_end"],   tz="UTC")
        ts = pd.Timestamp(win["train_start"], tz="UTC")

        train_df = raw[(raw.index >= ts) & (raw.index <= te)].copy()
        val_df   = raw[(raw.index >= vs) & (raw.index <= ve)].copy()

        if len(train_df) < 500 or len(val_df) < 100:
            print(f"[walk_forward] Window {i+1}: insufficient data, skipping")
            continue

        try:
            train_feat = add_features(train_df)
            val_feat   = add_features(val_df)

            X_train, idx_train = get_hmm_feature_matrix(train_feat)
            X_val,   idx_val   = get_hmm_feature_matrix(val_feat)

            # Retrain HMM fresh on this window's train data
            model = XAUUSDRegimeModel(n_states=p.get("n_states", 3))
            model.fit(X_train)

            def _signals(feat, X, idx):
                labels    = model.regime_labels(X)
                bull_prob = model.bull_probability(X)
                bear_prob = model.bear_probability(X)
                sig = generate_signals(feat, labels, bull_prob, bear_prob, idx)
                return compute_stops(sig, 1.5, 3.0)

            val_sig = _signals(val_feat, X_val, idx_val)
            n_sigs  = val_sig["signal_long"].sum() + val_sig["signal_short"].sum()

            if n_sigs < 5:
                print(f"[walk_forward] Window {i+1}: too few signals ({n_sigs}), skipping")
                continue

            metrics, _ = run_backtest(val_sig, strategy_name, period_label=f"wf_{i+1}")
            metrics["window"] = i + 1
            metrics["train_period"] = f"{win['train_start']} to {win['train_end']}"
            metrics["val_period"]   = f"{win['val_start']} to {win['val_end']}"
            results.append(metrics)
            print(f"[walk_forward] Window {i+1}: Sharpe={metrics.get('sharpe_ratio',0):.3f} | Return={metrics.get('annualized_return',0)*100:.1f}%")

        except Exception as e:
            print(f"[walk_forward] Window {i+1} failed: {e}")
            continue

    if not results:
        return {"available": False, "error": "No valid walk-forward windows"}

    returns = [r.get("annualized_return", 0) for r in results]
    sharpes = [r.get("sharpe_ratio", 0) for r in results]

    return {
        "available":    True,
        "n_windows":    len(results),
        "mean_return":  round(float(np.mean(returns)), 4),
        "std_return":   round(float(np.std(returns)), 4),
        "min_return":   round(float(np.min(returns)), 4),
        "max_return":   round(float(np.max(returns)), 4),
        "mean_sharpe":  round(float(np.mean(sharpes)), 4),
        "std_sharpe":   round(float(np.std(sharpes)), 4),
        "pct_positive": round(float(np.mean([r > 0 for r in returns])), 4),
        "windows":      results,
    }
