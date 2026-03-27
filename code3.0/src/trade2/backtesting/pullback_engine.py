"""
backtesting/pullback_engine.py - 3-TF pullback entry engine.

Runs bar-by-bar on 1M OHLCV bars. Reads 5M signals (forward-filled onto 1M)
and executes adaptive tiered limit orders whose depth is scaled by the 1H
regime label and the 5M bar's volume ratio.

Required df_1m columns (from forward_fill_5m_to_1m):
    Open, High, Low, Close               -- 1M OHLCV
    new_signal_long, new_signal_short    -- onset markers
    signal_long_5m, signal_short_5m     -- direction (forward-filled)
    signal_close_5m                      -- 5M bar close at signal time
    atr_5m                               -- 5M ATR at signal time
    vol_5m, vol_avg_5m                   -- for volume ratio
    regime                               -- 1H regime label (forward-filled)
    exit_long_5m, exit_short_5m         -- regime-flip exit signals
    position_size_long_5m, position_size_short_5m
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


def _get_tier_mults(
    regime: str,
    pb_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """Return (tier1_mult, tier2_mult) for the given 1H regime label."""
    r = str(regime).lower()
    if "bull" in r:
        t1 = pb_cfg["bull_tier1_mult"]
        t2 = pb_cfg["bull_tier2_mult"]
    elif "bear" in r:
        t1 = pb_cfg["bear_tier1_mult"]
        t2 = pb_cfg["bear_tier2_mult"]
    else:  # sideways / neutral
        t1 = pb_cfg["neutral_tier1_mult"]
        t2 = pb_cfg["neutral_tier2_mult"]
    # Enforce constraint: tier2 must be shallower (smaller offset from close)
    t2 = min(t2, t1 * 0.9)
    return t1, t2


def _compute_limits(
    direction: str,
    signal_close: float,
    atr_5m: float,
    vol_5m: float,
    vol_avg_5m: float,
    regime: str,
    pb_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Compute tier1_limit and tier2_limit for a pending order.

    vol_factor scales tier depth: high-volume bars expect deeper pullbacks.
    """
    t1_mult, t2_mult = _get_tier_mults(regime, pb_cfg)
    vol_ratio  = vol_5m / vol_avg_5m if vol_avg_5m > 0 else 1.0
    vol_factor = float(np.clip(vol_ratio, 0.5, 2.0))

    offset_t1 = t1_mult * atr_5m * vol_factor
    offset_t2 = t2_mult * atr_5m * vol_factor

    if direction == "long":
        tier1 = signal_close - offset_t1   # deeper dip (better price)
        tier2 = signal_close - offset_t2   # shallower dip (easier fill)
    else:  # short
        tier1 = signal_close + offset_t1
        tier2 = signal_close + offset_t2

    return tier1, tier2


def simulate(
    df_1m: pd.DataFrame,
    pb_cfg: Dict[str, Any],
    risk_cfg: Dict[str, Any],
    init_cash: float,
    commission_rt: float,
    slippage_arr: np.ndarray,
    contract_size_oz: float = 100.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Bar-by-bar 1M simulation with tiered pullback entries.

    Args:
        df_1m:          1M DataFrame with all forward-filled columns
        pb_cfg:         config['pullback'] dict
        risk_cfg:       config['risk'] dict
        init_cash:      starting capital
        commission_rt:  round-trip commission fraction
        slippage_arr:   per-bar slippage fractions (len = len(df_1m))
        contract_size_oz: oz per lot (default 100 for XAUUSD)

    Returns:
        equity_curve: pd.Series indexed like df_1m
        trades_df:    pd.DataFrame one row per completed trade
    """
    opens  = df_1m["Open"].values.astype(float)
    highs  = df_1m["High"].values.astype(float)
    lows   = df_1m["Low"].values.astype(float)
    closes = df_1m["Close"].values.astype(float)

    new_long  = df_1m["new_signal_long"].values.astype(int)
    new_short = df_1m["new_signal_short"].values.astype(int)
    sig_long  = df_1m["signal_long_5m"].values.astype(int)
    sig_short = df_1m["signal_short_5m"].values.astype(int)
    exit_long  = df_1m["exit_long_5m"].values.astype(int)
    exit_short = df_1m["exit_short_5m"].values.astype(int)

    signal_close_5m = df_1m["signal_close_5m"].values.astype(float)
    atr_5m_vals     = df_1m["atr_5m"].values.astype(float)
    vol_5m_vals     = df_1m["vol_5m"].values.astype(float)
    vol_avg_5m_vals = df_1m["vol_avg_5m"].values.astype(float)
    ps_long         = df_1m["position_size_long_5m"].values.astype(float)
    ps_short        = df_1m["position_size_short_5m"].values.astype(float)
    regimes         = df_1m["regime"].values

    max_wait     = int(pb_cfg["max_wait_bars"])
    runaway_mult = float(pb_cfg["runaway_atr_mult"])
    atr_stop_mult  = float(risk_cfg["atr_stop_mult"])
    atr_tp_mult    = float(risk_cfg["atr_tp_mult"])
    trail_mult     = float(risk_cfg["trailing_atr_mult"])
    trail_enabled  = bool(risk_cfg["trailing_enabled"])
    be_trigger     = float(risk_cfg["break_even_atr_trigger"])
    base_alloc     = float(risk_cfg["base_allocation_frac"])

    n = len(df_1m)
    equity_arr = np.zeros(n, dtype=float)
    trades: List[Dict] = []
    cash = float(init_cash)

    # --- Position state ---
    in_pos    = False
    direction = None
    entry_bar = -1
    entry_px  = 0.0
    frozen_sl = 0.0
    frozen_tp = 0.0
    pos_val   = 0.0
    n_units   = 0.0
    be_active = False
    entry_atr = 0.0
    hwm       = 0.0
    tier_used = 0

    # --- Pending order state ---
    pending: Optional[Dict] = None

    for i in range(n):

        # ================================================================
        # 1. Manage open position (exits)
        # ================================================================
        if in_pos:
            # Update trailing stop
            if trail_enabled and trail_mult > 0 and entry_atr > 0:
                if direction == "long":
                    hwm = max(hwm, closes[i])
                    new_sl = hwm - trail_mult * atr_5m_vals[i]
                    if new_sl > frozen_sl:
                        frozen_sl = new_sl
                else:
                    hwm = min(hwm, closes[i])
                    new_sl = hwm + trail_mult * atr_5m_vals[i]
                    if new_sl < frozen_sl:
                        frozen_sl = new_sl

            # Break-even stop
            if be_trigger > 0 and not be_active and entry_atr > 0:
                trigger_dist = be_trigger * entry_atr
                if direction == "long" and highs[i] >= entry_px + trigger_dist:
                    frozen_sl = max(frozen_sl, entry_px)
                    be_active = True
                elif direction == "short" and lows[i] <= entry_px - trigger_dist:
                    frozen_sl = min(frozen_sl, entry_px)
                    be_active = True

            # Check exits: SL > TP > regime flip
            reason  = None
            exit_px = 0.0

            if direction == "long":
                if lows[i] <= frozen_sl:
                    reason  = "sl"
                    exit_px = frozen_sl * (1.0 - slippage_arr[i])
                elif highs[i] >= frozen_tp:
                    reason  = "tp"
                    exit_px = frozen_tp * (1.0 - slippage_arr[i])
                elif exit_long[i] == 1 and i > entry_bar:
                    reason  = "signal"
                    exit_px = closes[i] * (1.0 - slippage_arr[i])
            else:
                if highs[i] >= frozen_sl:
                    reason  = "sl"
                    exit_px = frozen_sl * (1.0 + slippage_arr[i])
                elif lows[i] <= frozen_tp:
                    reason  = "tp"
                    exit_px = frozen_tp * (1.0 + slippage_arr[i])
                elif exit_short[i] == 1 and i > entry_bar:
                    reason  = "signal"
                    exit_px = closes[i] * (1.0 + slippage_arr[i])

            if reason is not None:
                if direction == "long":
                    raw_pnl = (exit_px - entry_px) * n_units
                else:
                    raw_pnl = (entry_px - exit_px) * n_units
                cost    = commission_rt * pos_val
                net_pnl = raw_pnl - cost
                cash   += net_pnl
                equity_arr[i] = cash
                lots = n_units / contract_size_oz if contract_size_oz > 0 else 0.0
                trades.append({
                    "entry_time":    df_1m.index[entry_bar],
                    "exit_time":     df_1m.index[i],
                    "direction":     direction,
                    "entry_price":   round(entry_px, 5),
                    "exit_price":    round(exit_px,  5),
                    "sl":            round(frozen_sl, 5),
                    "tp":            round(frozen_tp, 5),
                    "size":          round(pos_val, 2),
                    "lots":          round(lots, 4),
                    "pnl":           round(net_pnl, 4),
                    "duration_bars": i - entry_bar,
                    "exit_reason":   reason,
                    "tier":          tier_used,
                })
                in_pos    = False
                direction = None
                pending   = None
            else:
                equity_arr[i] = cash + (
                    (closes[i] - entry_px) * n_units if direction == "long"
                    else (entry_px - closes[i]) * n_units
                )

        else:
            equity_arr[i] = cash

        # ================================================================
        # 2. Handle pending order (if no open position)
        # ================================================================
        if not in_pos and pending is not None:
            pd_dir = pending["dir"]

            # 2a. Same-direction new signal: refresh levels, reset counter
            if (pd_dir == "long" and new_long[i]) or (pd_dir == "short" and new_short[i]):
                t1, t2 = _compute_limits(
                    pd_dir, signal_close_5m[i], atr_5m_vals[i],
                    vol_5m_vals[i], vol_avg_5m_vals[i], regimes[i], pb_cfg,
                )
                pending = {
                    "dir": pd_dir, "tier1": t1, "tier2": t2,
                    "bar_count": 0, "signal_close": signal_close_5m[i],
                    "atr": atr_5m_vals[i],
                    "ps": ps_long[i] if pd_dir == "long" else ps_short[i],
                }

            else:
                # 2b. Check invalidation
                atr_now = atr_5m_vals[i]
                sc      = pending["signal_close"]
                invalidated = False

                regime_now = str(regimes[i]).lower()
                if pd_dir == "long" and "bear" in regime_now:
                    invalidated = True
                elif pd_dir == "short" and "bull" in regime_now:
                    invalidated = True

                if not invalidated:
                    if pd_dir == "long" and closes[i] > sc + runaway_mult * atr_now:
                        invalidated = True
                    elif pd_dir == "short" and closes[i] < sc - runaway_mult * atr_now:
                        invalidated = True

                if not invalidated:
                    if pd_dir == "long" and new_short[i]:
                        invalidated = True
                    elif pd_dir == "short" and new_long[i]:
                        invalidated = True

                if invalidated:
                    pending = None

                else:
                    # 2c. Try fills (tier1 first)
                    filled   = False
                    fill_px  = 0.0
                    tier_hit = 0

                    if pd_dir == "long":
                        if lows[i] <= pending["tier1"]:
                            fill_px  = pending["tier1"]
                            tier_hit = 1
                            filled   = True
                        elif lows[i] <= pending["tier2"]:
                            fill_px  = pending["tier2"]
                            tier_hit = 2
                            filled   = True
                    else:
                        if highs[i] >= pending["tier1"]:
                            fill_px  = pending["tier1"]
                            tier_hit = 1
                            filled   = True
                        elif highs[i] >= pending["tier2"]:
                            fill_px  = pending["tier2"]
                            tier_hit = 2
                            filled   = True

                    if filled:
                        # Apply slippage to fill price
                        if pd_dir == "long":
                            fill_px = fill_px * (1.0 + slippage_arr[i])
                        else:
                            fill_px = fill_px * (1.0 - slippage_arr[i])

                        fill_atr = pending["atr"]
                        if pd_dir == "long":
                            sl = fill_px - atr_stop_mult * fill_atr
                            tp = fill_px + atr_tp_mult  * fill_atr
                        else:
                            sl = fill_px + atr_stop_mult * fill_atr
                            tp = fill_px - atr_tp_mult  * fill_atr

                        pos_val   = cash * base_alloc * pending["ps"]
                        n_units   = pos_val / fill_px if fill_px > 0 else 0.0
                        in_pos    = True
                        direction = pd_dir
                        entry_bar = i
                        entry_px  = fill_px
                        frozen_sl = sl
                        frozen_tp = tp
                        entry_atr = fill_atr
                        hwm       = fill_px
                        be_active = False
                        tier_used = tier_hit
                        pending   = None

                    else:
                        pending["bar_count"] += 1
                        if pending["bar_count"] >= max_wait:
                            pending = None

        # ================================================================
        # 3. Check for new signal (only if no position and no pending)
        # ================================================================
        if not in_pos and pending is None:
            signal_dir = None
            ps_val     = 0.0
            if new_long[i] and not new_short[i]:
                signal_dir = "long"
                ps_val     = ps_long[i]
            elif new_short[i] and not new_long[i]:
                signal_dir = "short"
                ps_val     = ps_short[i]

            if signal_dir is not None:
                t1, t2 = _compute_limits(
                    signal_dir, signal_close_5m[i], atr_5m_vals[i],
                    vol_5m_vals[i], vol_avg_5m_vals[i], regimes[i], pb_cfg,
                )
                pending = {
                    "dir": signal_dir, "tier1": t1, "tier2": t2,
                    "bar_count": 0, "signal_close": signal_close_5m[i],
                    "atr": atr_5m_vals[i], "ps": ps_val,
                }

    # Close any open position at end of data
    if in_pos:
        i = n - 1
        if direction == "long":
            exit_px = closes[i] * (1.0 - slippage_arr[i])
            raw_pnl = (exit_px - entry_px) * n_units
        else:
            exit_px = closes[i] * (1.0 + slippage_arr[i])
            raw_pnl = (entry_px - exit_px) * n_units
        cost    = commission_rt * pos_val
        net_pnl = raw_pnl - cost
        cash   += net_pnl
        equity_arr[i] = cash
        lots = n_units / contract_size_oz if contract_size_oz > 0 else 0.0
        trades.append({
            "entry_time":    df_1m.index[entry_bar],
            "exit_time":     df_1m.index[i],
            "direction":     direction,
            "entry_price":   round(entry_px, 5),
            "exit_price":    round(exit_px,  5),
            "sl":            round(frozen_sl, 5),
            "tp":            round(frozen_tp, 5),
            "size":          round(pos_val, 2),
            "lots":          round(lots, 4),
            "pnl":           round(net_pnl, 4),
            "duration_bars": i - entry_bar,
            "exit_reason":   "end_of_data",
            "tier":          tier_used,
        })

    equity_curve = pd.Series(equity_arr, index=df_1m.index, name="equity")
    trades_df = (
        pd.DataFrame(trades) if trades else
        pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction", "entry_price", "exit_price",
            "sl", "tp", "size", "lots", "pnl", "duration_bars", "exit_reason", "tier",
        ])
    )
    return equity_curve, trades_df
