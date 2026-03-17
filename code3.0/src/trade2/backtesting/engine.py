"""
backtesting/engine.py - Custom event-driven backtest engine.
Simulates trades bar-by-bar with real ATR-based SL/TP execution and
confidence-based variable position sizing. No vectorbt dependency.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from trade2.backtesting.costs import compute_slippage_array, doubled_costs
from trade2.backtesting.metrics import compute_metrics, compute_random_baseline, format_report


def _simulate_trades(
    df: pd.DataFrame,
    init_cash: float,
    base_allocation_frac: float,
    slippage: np.ndarray,
    commission_rt: float,
    max_hold_bars: int,
    be_atr_trigger: float = 0.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Bar-by-bar event-driven simulation with frozen ATR-based SL/TP
    and confidence-based variable position sizing.

    Signal at bar i  -> entry at bar i+1 Open (next-bar execution).
    SL/TP levels are frozen from the signal bar and never updated.
    One position at a time; conflicting long+short signals are skipped.

    Required df columns:
        Open, High, Low, Close
        signal_long, signal_short, exit_long, exit_short
        stop_long, stop_short, tp_long, tp_short
        position_size_long, position_size_short

    Returns:
        equity_curve : pd.Series  (bar-by-bar mark-to-market portfolio value)
        trades_df    : pd.DataFrame  (one row per completed trade)
    """
    opens  = df["Open"].values.astype(float)
    highs  = df["High"].values.astype(float)
    lows   = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)

    sig_long   = df["signal_long"].values.astype(int)
    sig_short  = df["signal_short"].values.astype(int)
    exit_long  = df["exit_long"].values.astype(int)
    exit_short = df["exit_short"].values.astype(int)
    stop_long  = df["stop_long"].values.astype(float)
    stop_short = df["stop_short"].values.astype(float)
    tp_long    = df["tp_long"].values.astype(float)
    tp_short   = df["tp_short"].values.astype(float)
    ps_long    = df["position_size_long"].values.astype(float)
    ps_short   = df["position_size_short"].values.astype(float)

    # Trailing stop: per-bar ATR multiplier (0 = no trailing)
    trail_long  = df["trailing_atr_mult_long"].values.astype(float)  if "trailing_atr_mult_long"  in df.columns else np.zeros(len(df))
    trail_short = df["trailing_atr_mult_short"].values.astype(float) if "trailing_atr_mult_short" in df.columns else np.zeros(len(df))
    atr_vals    = (df["atr_1h"].values if "atr_1h" in df.columns
                   else df["atr_14"].values if "atr_14" in df.columns
                   else np.ones(len(df))).astype(float)

    n          = len(df)
    equity_arr = np.zeros(n, dtype=float)
    trades: List[Dict] = []

    cash = float(init_cash)

    # Active position state
    in_pos      = False
    direction   = None    # 'long' | 'short'
    entry_bar   = -1
    entry_px    = 0.0
    frozen_sl   = 0.0
    frozen_tp   = 0.0
    pos_val     = 0.0     # dollar value committed
    n_units     = 0.0     # units held
    be_active   = False   # break-even stop already triggered
    entry_atr   = 0.0     # ATR at entry bar (for BE trigger distance)

    # Pending entry: signal at bar i fires at bar i+1 open
    pend_dir = None
    pend_sl  = 0.0
    pend_tp  = 0.0
    pend_ps  = 0.0      # position_size multiplier

    for i in range(n):

        # --- Execute pending entry at this bar's open ---
        if pend_dir is not None and not in_pos:
            if pend_dir == "long":
                entry_px = opens[i] * (1.0 + slippage[i])
            else:
                entry_px = opens[i] * (1.0 - slippage[i])
            pos_val   = cash * base_allocation_frac * pend_ps
            n_units   = pos_val / entry_px if entry_px > 0.0 else 0.0
            frozen_sl = pend_sl
            frozen_tp = pend_tp
            direction = pend_dir
            entry_bar = i
            entry_atr = atr_vals[i]
            in_pos    = True
            be_active = False
            pend_dir  = None

        # --- Mark-to-market equity ---
        if in_pos:
            if direction == "long":
                equity_arr[i] = cash + (closes[i] - entry_px) * n_units
            else:
                equity_arr[i] = cash + (entry_px - closes[i]) * n_units
        else:
            equity_arr[i] = cash

        # --- Update trailing stop (before SL check) ---
        if in_pos:
            if direction == "long" and trail_long[i] > 0:
                new_sl = closes[i] - trail_long[i] * atr_vals[i]
                if new_sl > frozen_sl:
                    frozen_sl = new_sl
            elif direction == "short" and trail_short[i] > 0:
                new_sl = closes[i] + trail_short[i] * atr_vals[i]
                if new_sl < frozen_sl:
                    frozen_sl = new_sl

        # --- Break-even stop: move SL to entry when price moves be_atr_trigger * ATR in favor ---
        if in_pos and be_atr_trigger > 0 and not be_active and entry_atr > 0:
            trigger_dist = be_atr_trigger * entry_atr
            if direction == "long" and highs[i] >= entry_px + trigger_dist:
                frozen_sl = max(frozen_sl, entry_px)
                be_active = True
            elif direction == "short" and lows[i] <= entry_px - trigger_dist:
                frozen_sl = min(frozen_sl, entry_px)
                be_active = True

        # --- Check SL / TP / signal exit / timeout ---
        if in_pos:
            reason  = None
            exit_px = 0.0

            if direction == "long":
                if lows[i] <= frozen_sl:
                    reason  = "sl"
                    exit_px = frozen_sl * (1.0 - slippage[i])
                elif highs[i] >= frozen_tp:
                    reason  = "tp"
                    exit_px = frozen_tp * (1.0 - slippage[i])
                elif exit_long[i] == 1 and i > entry_bar:
                    reason  = "signal"
                    exit_px = closes[i] * (1.0 - slippage[i])
                elif max_hold_bars > 0 and (i - entry_bar) >= max_hold_bars:
                    reason  = "timeout"
                    exit_px = closes[i] * (1.0 - slippage[i])
            else:  # short
                if highs[i] >= frozen_sl:
                    reason  = "sl"
                    exit_px = frozen_sl * (1.0 + slippage[i])
                elif lows[i] <= frozen_tp:
                    reason  = "tp"
                    exit_px = frozen_tp * (1.0 + slippage[i])
                elif exit_short[i] == 1 and i > entry_bar:
                    reason  = "signal"
                    exit_px = closes[i] * (1.0 + slippage[i])
                elif max_hold_bars > 0 and (i - entry_bar) >= max_hold_bars:
                    reason  = "timeout"
                    exit_px = closes[i] * (1.0 + slippage[i])

            if reason is not None:
                if direction == "long":
                    raw_pnl = (exit_px - entry_px) * n_units
                else:
                    raw_pnl = (entry_px - exit_px) * n_units
                cost      = commission_rt * pos_val
                net_pnl   = raw_pnl - cost
                cash     += net_pnl
                equity_arr[i] = cash  # lock in realized equity

                trades.append({
                    "entry_time":    df.index[entry_bar],
                    "exit_time":     df.index[i],
                    "direction":     direction,
                    "entry_price":   round(entry_px,  5),
                    "exit_price":    round(exit_px,   5),
                    "sl":            round(frozen_sl,  5),
                    "tp":            round(frozen_tp,  5),
                    "size":          round(pos_val,    2),
                    "pnl":           round(net_pnl,    4),
                    "duration_bars": i - entry_bar,
                    "exit_reason":   reason,
                })
                in_pos    = False
                direction = None

        # --- Queue next entry from signal at bar i ---
        if not in_pos and pend_dir is None and i < n - 1:
            go_long  = sig_long[i]  == 1 and sig_short[i] == 0
            go_short = sig_short[i] == 1 and sig_long[i]  == 0

            if go_long:
                pend_dir = "long"
                pend_sl  = stop_long[i]
                pend_tp  = tp_long[i]
                pend_ps  = ps_long[i]
            elif go_short:
                pend_dir = "short"
                pend_sl  = stop_short[i]
                pend_tp  = tp_short[i]
                pend_ps  = ps_short[i]

    # --- Close any open position at end of data ---
    if in_pos:
        i = n - 1
        if direction == "long":
            exit_px = closes[i] * (1.0 - slippage[i])
            raw_pnl = (exit_px - entry_px) * n_units
        else:
            exit_px = closes[i] * (1.0 + slippage[i])
            raw_pnl = (entry_px - exit_px) * n_units
        cost      = commission_rt * pos_val
        net_pnl   = raw_pnl - cost
        cash     += net_pnl
        equity_arr[i] = cash
        trades.append({
            "entry_time":    df.index[entry_bar],
            "exit_time":     df.index[i],
            "direction":     direction,
            "entry_price":   round(entry_px,  5),
            "exit_price":    round(exit_px,   5),
            "sl":            round(frozen_sl,  5),
            "tp":            round(frozen_tp,  5),
            "size":          round(pos_val,    2),
            "pnl":           round(net_pnl,    4),
            "duration_bars": i - entry_bar,
            "exit_reason":   "end_of_data",
        })

    equity_curve = pd.Series(equity_arr, index=df.index, name="equity")
    trades_df = (
        pd.DataFrame(trades) if trades else
        pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction", "entry_price", "exit_price",
            "sl", "tp", "size", "pnl", "duration_bars", "exit_reason",
        ])
    )
    return equity_curve, trades_df


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    period_label: str = "test",
    config: Dict[str, Any] = None,
    backtests_dir: Path = None,
    freq: str = "1h",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run event-driven backtest using pre-computed signal + SL/TP columns in df.
    All parameters read from config -- no hardcoded defaults.

    Expected df columns: signal_long, signal_short, exit_long, exit_short,
        stop_long, stop_short, tp_long, tp_short,
        position_size_long, position_size_short

    Args:
        df:            Feature+signal DataFrame (with stop/tp columns from compute_stops)
        strategy_name: Name for result file
        period_label:  'train', 'val', 'test', or 'wf_N'
        config:        Config dict (all parameters sourced from here)
        backtests_dir: Where to save result JSON (optional)
        freq:          '1h' or '5min' -- controls bars_per_year

    Returns:
        (metrics_dict, trades_df)
    """
    cfg       = config or {}
    costs_cfg = cfg["costs"]
    risk_cfg  = cfg["risk"]
    hmm_cfg   = cfg["hmm"]

    init_cash       = cfg["backtest"]["init_cash"]
    commission      = costs_cfg["commission_rt"]
    max_hold_bars   = risk_cfg["max_hold_bars"]
    base_alloc      = risk_cfg["base_allocation_frac"]
    be_atr_trigger  = risk_cfg.get("break_even_atr_trigger", 0.0)

    # Scale max_hold_bars and bars_per_year to match the data frequency.
    # Config value is expressed in 1H-equivalent bars.
    _TF_SCALE = {
        "5min": 12, "5m": 12, "5M": 12,
        "15min": 4, "15m": 4, "15M": 4,
        "30min": 2, "30m": 2, "30M": 2,
        "1h": 1, "1H": 1, "4h": 0,  # 4H: scale=0 (same bars, different unit)
    }
    tf_scale = _TF_SCALE.get(freq, 1)
    if tf_scale > 1:
        max_hold_bars = max_hold_bars * tf_scale

    close = df["Close"].astype(float)

    n_longs  = int(df["signal_long"].sum())
    n_shorts = int(df["signal_short"].sum())
    if n_longs == 0 and n_shorts == 0:
        print(f"[engine] WARNING: No signals for {period_label}")
        return {
            "annualized_return": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "total_trades": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
        }, pd.DataFrame()

    slippage = compute_slippage_array(close, cfg).values

    bars_per_year = 252 * 24 * max(tf_scale, 1)

    equity, trades_df = _simulate_trades(
        df                   = df,
        init_cash            = init_cash,
        base_allocation_frac = base_alloc,
        slippage             = slippage,
        commission_rt        = commission,
        max_hold_bars        = max_hold_bars,
        be_atr_trigger       = be_atr_trigger,
    )

    trade_records = None
    if len(trades_df) > 0:
        trade_records = trades_df[["pnl", "duration_bars"]].copy()

    bh_equity = init_cash * (close / close.iloc[0])
    metrics   = compute_metrics(
        equity_curve     = equity,
        trades           = trade_records,
        benchmark_equity = bh_equity,
        bars_per_year    = bars_per_year,
    )

    # Exit reason breakdown
    if len(trades_df) > 0 and "exit_reason" in trades_df.columns:
        metrics["exit_reasons"] = trades_df["exit_reason"].value_counts().to_dict()

    if metrics.get("total_trades", 0) > 0:
        baseline = compute_random_baseline(
            close=close, n_trades=metrics["total_trades"], bars_per_year=bars_per_year
        )
        metrics["random_baseline"] = baseline
        beats = metrics.get("sharpe_ratio", 0) > baseline["random_p95_sharpe"]
        metrics["beats_random_baseline"] = beats
        print(
            f"[engine] Random baseline p95 Sharpe: {baseline['random_p95_sharpe']:.3f}"
            f" | Strategy: {metrics.get('sharpe_ratio', 0):.3f} | Beats: {beats}"
        )

    print(format_report(metrics, f"{strategy_name} [{period_label}]"))

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
                    "spread_pips":   costs_cfg["spread_pips"],
                    "slippage_pips": costs_cfg["slippage_pips"],
                    "commission_rt": costs_cfg["commission_rt"],
                },
            }, f, indent=2)
        print(f"[engine] Results saved to {result_path}")

    return metrics, trades_df


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
    raw_regime_path: Path,
    backtests_dir: Path = None,
    freq: str = None,
    raw_signal_path: Path = None,
) -> Dict[str, Any]:
    """
    Walk-forward validation. Each window independently:
    loads data, builds features, RETRAINS HMM, generates signals, backtests.

    When raw_signal_path is provided AND strategy.mode == multi_tf:
        - Regime features built from regime TF (1H/4H) bars
        - Signals generated on signal TF (5M) bars with regime forward-filled
        - Backtest runs on signal TF bars (freq="5min")
    Otherwise falls back to regime-TF-only signal path.

    Args:
        strategy_name:    Name for result files
        config:           Full config dict
        raw_regime_path:  Path to raw CSV for the regime timeframe (1H, 4H, etc.)
        backtests_dir:    Where to save per-window results
        freq:             Backtest frequency override (auto-detected from config if None)
        raw_signal_path:  Path to 5M raw CSV for multi_tf signal generation (optional)

    Returns:
        Summary dict with per-window metrics and aggregate stats.
    """
    from trade2.data.loader import load_raw
    from trade2.features.builder import add_1h_features, add_5m_features
    from trade2.features.hmm_features import get_hmm_feature_matrix
    from trade2.models.hmm import XAUUSDRegimeModel
    from trade2.signals.generator import generate_signals, compute_stops, compute_stops_regime_aware
    from trade2.signals.regime import forward_fill_1h_regime
    from trade2.signals.router import route_signals, apply_tv_signal_filter, ffill_tv_cols_to_5m

    windows = config.get("walk_forward", {}).get("windows", [])
    if not windows:
        return {"available": False, "error": "No walk-forward windows defined in config"}

    strategy_mode_pipeline = config.get("strategy", {}).get("mode", "single_tf")
    use_5m = (raw_signal_path is not None) and (strategy_mode_pipeline == "multi_tf")

    _TF_TO_FREQ_WF = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"}
    if freq is None:
        if use_5m:
            signal_tf = config.get("strategy", {}).get("signal_timeframe", "5M")
            freq = _TF_TO_FREQ_WF.get(signal_tf, "5min")
        else:
            regime_tf = config.get("strategy", {}).get("regime_timeframe", "1H")
            freq = _TF_TO_FREQ_WF.get(regime_tf, "1h")

    raw_reg  = load_raw(raw_regime_path)
    raw_sig  = load_raw(Path(raw_signal_path)) if use_5m else None
    hmm_cfg  = config.get("hmm", {})
    results  = []

    for i, win in enumerate(windows):
        te = pd.Timestamp(win["train_end"],   tz="UTC")
        vs = pd.Timestamp(win["val_start"],   tz="UTC")
        ve = pd.Timestamp(win["val_end"],     tz="UTC")
        ts = pd.Timestamp(win["train_start"], tz="UTC")

        train_df = raw_reg[(raw_reg.index >= ts) & (raw_reg.index <= te)].copy()
        val_df   = raw_reg[(raw_reg.index >= vs) & (raw_reg.index <= ve)].copy()

        if len(train_df) < 500 or len(val_df) < 100:
            print(f"[walk_forward] Window {i+1}: insufficient regime data, skipping")
            continue

        try:
            train_feat = add_1h_features(train_df, config)
            val_feat   = add_1h_features(val_df,   config)

            X_train, idx_train = get_hmm_feature_matrix(train_feat, config)
            X_val,   idx_val   = get_hmm_feature_matrix(val_feat, config)

            model = XAUUSDRegimeModel(
                n_states    = hmm_cfg.get("n_states", 3),
                random_seed = hmm_cfg.get("random_seed", 42),
            )
            model.fit(X_train)

            wf_strategy_mode = config.get("strategies", {}).get("mode", "legacy")

            if use_5m:
                # Multi-TF path: build 5M signal bars with regime forward-filled
                val_5m = raw_sig[(raw_sig.index >= vs) & (raw_sig.index <= ve)].copy()
                if len(val_5m) < 100:
                    print(f"[walk_forward] Window {i+1}: insufficient 5M data, skipping")
                    continue
                val_5m_feat = add_5m_features(val_5m, config)
                sideways_prob = model.sideways_probability(X_val) if hasattr(model, "sideways_probability") else None
                val_sig_df = forward_fill_1h_regime(
                    val_5m_feat,
                    hmm_labels        = model.regime_labels(X_val),
                    hmm_bull_prob     = model.bull_probability(X_val),
                    hmm_bear_prob     = model.bear_probability(X_val),
                    hmm_index         = idx_val,
                    atr_1h            = val_feat["atr_14"].rename(None),
                    hma_rising        = val_feat["hma_rising"].rename(None),
                    price_above_hma   = val_feat["price_above_hma"].rename(None),
                    hmm_sideways_prob = sideways_prob,
                )
                val_sig_df = ffill_tv_cols_to_5m(val_sig_df, val_feat)
                if wf_strategy_mode == "regime_specialized":
                    val_sig = route_signals(val_sig_df, config)
                    val_sig = apply_tv_signal_filter(val_sig, config)
                    val_sig = compute_stops_regime_aware(val_sig, config)
                else:
                    val_sig = generate_signals(val_sig_df, config)
                    val_sig = compute_stops(val_sig, config["risk"]["atr_stop_mult"], config["risk"]["atr_tp_mult"])
            elif wf_strategy_mode == "regime_specialized":
                val_sig = route_signals(
                    val_feat, config,
                    hmm_labels    = model.regime_labels(X_val),
                    hmm_bull_prob = model.bull_probability(X_val),
                    hmm_bear_prob = model.bear_probability(X_val),
                    hmm_index     = idx_val,
                )
                val_sig = compute_stops_regime_aware(val_sig, config)
            else:
                val_sig = generate_signals(
                    val_feat, config,
                    hmm_labels    = model.regime_labels(X_val),
                    hmm_bull_prob = model.bull_probability(X_val),
                    hmm_bear_prob = model.bear_probability(X_val),
                    hmm_index     = idx_val,
                )
                val_sig = compute_stops(
                    val_sig,
                    config["risk"]["atr_stop_mult"],
                    config["risk"]["atr_tp_mult"],
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
            print(
                f"[walk_forward] Window {i+1}:"
                f" Sharpe={metrics.get('sharpe_ratio',0):.3f}"
                f" | Return={metrics.get('annualized_return',0)*100:.1f}%"
            )

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
