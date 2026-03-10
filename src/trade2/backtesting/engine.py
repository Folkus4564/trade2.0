"""
backtesting/engine.py - vectorbt backtest engine supporting 1H and 5M frequencies.
No module-level mkdir. Artefact paths passed explicitly.
"""

import json
import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from trade2.backtesting.costs import compute_slippage, doubled_costs
from trade2.backtesting.metrics import compute_metrics, compute_random_baseline, format_report


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    period_label: str = "test",
    config: Dict[str, Any] = None,
    backtests_dir: Path = None,
    init_cash: float = 100_000.0,
    size_pct: float = 0.95,
    freq: str = "1h",
) -> Tuple[Dict[str, Any], Any]:
    """
    Run vectorbt backtest using signal columns in df.

    Expected columns: signal_long, signal_short, exit_long, exit_short

    Args:
        df:            Feature+signal DataFrame
        strategy_name: Name for result file
        period_label:  'train', 'val', 'test', or 'wf_N'
        config:        Config dict (for costs)
        backtests_dir: Where to save result JSON (optional)
        freq:          '1h' or '5min' — controls bars_per_year and vectorbt freq
        size_pct:      Fraction of portfolio per trade

    Returns:
        (metrics_dict, vbt.Portfolio)
    """
    cfg = config or {}
    costs_cfg    = cfg.get("costs", {})
    commission   = costs_cfg.get("commission_rt", 0.0002)

    close = df["Close"].astype(float)

    long_entries  = df["signal_long"].astype(bool)
    long_exits    = df["exit_long"].astype(bool)
    short_entries = df["signal_short"].astype(bool)
    short_exits   = df["exit_short"].astype(bool)

    conflict      = long_entries & short_entries
    long_entries  = long_entries  & ~conflict
    short_entries = short_entries & ~conflict

    if long_entries.sum() == 0 and short_entries.sum() == 0:
        print(f"[engine] WARNING: No signals for {period_label}")
        return {
            "annualized_return": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "total_trades": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
        }, None

    slippage    = compute_slippage(close, cfg) if cfg else 0.001
    trade_value = init_cash * size_pct

    # bars_per_year depends on frequency
    if freq in ("5min", "5m", "5M"):
        bars_per_year = 252 * 24 * 12
        vbt_freq      = "5min"
    else:
        bars_per_year = 252 * 24
        vbt_freq      = "1h"

    pf = vbt.Portfolio.from_signals(
        close         = close,
        entries       = long_entries,
        exits         = long_exits,
        short_entries = short_entries,
        short_exits   = short_exits,
        init_cash     = init_cash,
        fees          = commission / 2,
        slippage      = slippage,
        size          = trade_value,
        size_type     = "value",
        freq          = vbt_freq,
    )

    equity    = pf.value()
    trades_df = pf.trades.records_readable

    trade_records = None
    if len(trades_df) > 0:
        trade_records = pd.DataFrame({
            "pnl":           trades_df.get("PnL", pd.Series(dtype=float)),
            "duration_bars": trades_df.get("Duration", pd.Series(dtype=float)),
        })

    bh_equity = init_cash * (close / close.iloc[0])

    metrics = compute_metrics(
        equity_curve     = equity,
        trades           = trade_records,
        benchmark_equity = bh_equity,
        bars_per_year    = bars_per_year,
    )

    if metrics.get("total_trades", 0) > 0:
        baseline = compute_random_baseline(
            close=close, n_trades=metrics["total_trades"], bars_per_year=bars_per_year
        )
        metrics["random_baseline"] = baseline
        beats = metrics.get("sharpe_ratio", 0) > baseline["random_p95_sharpe"]
        metrics["beats_random_baseline"] = beats
        print(f"[engine] Random baseline p95 Sharpe: {baseline['random_p95_sharpe']:.3f} | Strategy: {metrics.get('sharpe_ratio', 0):.3f} | Beats: {beats}")

    print(format_report(metrics, f"{strategy_name} [{period_label}]"))

    # Save results JSON
    if backtests_dir is not None:
        backtests_dir = Path(backtests_dir)
        backtests_dir.mkdir(parents=True, exist_ok=True)
        result_path = backtests_dir / f"{strategy_name}_{period_label}_results.json"
        with open(result_path, "w") as f:
            json.dump({
                "strategy": strategy_name,
                "period":   period_label,
                "start":    str(df.index[0]),
                "end":      str(df.index[-1]),
                "n_bars":   len(df),
                "metrics":  metrics,
                "costs": {
                    "spread_pips":   costs_cfg.get("spread_pips", 3),
                    "slippage_pips": costs_cfg.get("slippage_pips", 1),
                    "commission_rt": commission,
                },
            }, f, indent=2)
        print(f"[engine] Results saved to {result_path}")

    return metrics, pf


def run_backtest_2x_costs(
    df: pd.DataFrame,
    strategy_name: str,
    period_label: str = "test",
    config: Dict[str, Any] = None,
    backtests_dir: Path = None,
    freq: str = "1h",
) -> Dict[str, Any]:
    """Run backtest with 2x costs for sensitivity testing."""
    cfg_2x = doubled_costs(config or {})
    metrics, _ = run_backtest(
        df, strategy_name, period_label + "_2xcost",
        config=cfg_2x, backtests_dir=backtests_dir, freq=freq,
    )
    return metrics


def run_walk_forward(
    strategy_name: str,
    config: Dict[str, Any],
    raw_1h_path: Path,
    backtests_dir: Path = None,
    freq: str = "1h",
) -> Dict[str, Any]:
    """
    Walk-forward validation. Each window independently:
    loads data, builds features, RETRAINS HMM, generates signals, backtests.

    Args:
        strategy_name: Name for result files
        config:        Full config dict
        raw_1h_path:   Path to raw 1H CSV (HMM is always trained on 1H)
        backtests_dir: Where to save per-window results
        freq:          Backtest frequency for bars_per_year

    Returns:
        Summary dict with per-window metrics and aggregate stats.
    """
    from trade2.data.loader import load_raw
    from trade2.features.builder import add_1h_features
    from trade2.features.hmm_features import get_hmm_feature_matrix
    from trade2.models.hmm import XAUUSDRegimeModel
    from trade2.signals.generator import generate_signals, compute_stops

    windows = config.get("walk_forward", {}).get("windows", [])
    if not windows:
        return {"available": False, "error": "No walk-forward windows defined in config"}

    raw = load_raw(raw_1h_path)
    hmm_cfg = config.get("hmm", {})
    results = []

    for i, win in enumerate(windows):
        te = pd.Timestamp(win["train_end"],   tz="UTC")
        vs = pd.Timestamp(win["val_start"],   tz="UTC")
        ve = pd.Timestamp(win["val_end"],     tz="UTC")
        ts = pd.Timestamp(win["train_start"], tz="UTC")

        train_df = raw[(raw.index >= ts) & (raw.index <= te)].copy()
        val_df   = raw[(raw.index >= vs) & (raw.index <= ve)].copy()

        if len(train_df) < 500 or len(val_df) < 100:
            print(f"[walk_forward] Window {i+1}: insufficient data, skipping")
            continue

        try:
            train_feat = add_1h_features(train_df, config)
            val_feat   = add_1h_features(val_df,   config)

            X_train, idx_train = get_hmm_feature_matrix(train_feat)
            X_val,   idx_val   = get_hmm_feature_matrix(val_feat)

            model = XAUUSDRegimeModel(
                n_states    = hmm_cfg.get("n_states", 3),
                random_seed = hmm_cfg.get("random_seed", 42),
            )
            model.fit(X_train)

            val_sig = generate_signals(
                val_feat, config,
                hmm_labels    = model.regime_labels(X_val),
                hmm_bull_prob = model.bull_probability(X_val),
                hmm_bear_prob = model.bear_probability(X_val),
                hmm_index     = idx_val,
            )
            val_sig = compute_stops(
                val_sig,
                config.get("risk", {}).get("atr_stop_mult", 1.5),
                config.get("risk", {}).get("atr_tp_mult",   3.0),
            )

            n_sigs = val_sig["signal_long"].sum() + val_sig["signal_short"].sum()
            if n_sigs < 5:
                print(f"[walk_forward] Window {i+1}: too few signals ({n_sigs}), skipping")
                continue

            metrics, _ = run_backtest(
                val_sig, strategy_name, f"wf_{i+1}",
                config=config, backtests_dir=backtests_dir, freq=freq,
            )
            metrics["window"]       = i + 1
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
