# Scalp 3TF Pullback Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a triple-timeframe XAUUSD scalping strategy — 1H HMM regime, 5M entry signals, 1M pullback entry execution — with adaptive volume-scaled, regime-aware tier limits optimized by Optuna.

**Architecture:** 1H HMM labels are forward-filled to 5M and 1M bars. The 5M signal generator fires direction signals + ATR values, which are forward-filled to 1M. A new `PullbackEngine` runs bar-by-bar on 1M OHLCV, placing tiered limit orders (tier1 = deeper, tier2 = shallower) whose depth is scaled by the 1H regime and the 5M bar's volume ratio. SL/TP are sized from 5M ATR at fill price.

**Tech Stack:** Python, pandas, numpy, Optuna, TA-Lib, trade2 package (code3.0)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `code3.0/configs/scalp_3tf.yaml` | Config overlay for 3-TF strategy |
| Create | `code3.0/src/trade2/backtesting/pullback_engine.py` | `PullbackEngine` — bar-by-bar 1M simulation with tiered limits |
| Modify | `code3.0/src/trade2/signals/regime.py` | Add `forward_fill_5m_to_1m()` |
| Modify | `code3.0/src/trade2/app/run_pipeline.py` | Add 3-TF data loading + PullbackEngine orchestration |
| Modify | `code3.0/src/trade2/optimization/optimizer.py` | Add pullback trial function + search space handling |

---

## Task 1: Create `scalp_3tf.yaml`

**Files:**
- Create: `code3.0/configs/scalp_3tf.yaml`

- [ ] **Step 1: Write the config file**

```yaml
# ============================================================
#  trade2 - Scalp 3TF Pullback Strategy
#  1H HMM regime / 5M signals / 1M pullback entry
# ============================================================

strategy:
  name: xauusd_scalp_3tf
  mode: multi_tf
  regime_timeframe: 1H
  signal_timeframe: 5M
  entry_timeframe: 1M      # triggers 3-TF path in run_pipeline

data:
  raw_1h_csv: data/raw/XAUUSD_1H_2019_2026_full.csv
  raw_5m_csv: data/raw/XAUUSD_5M_2019_2026.csv
  raw_1m_csv: code3.0/data/raw/XAUUSD_1M_2019_2026.csv
  missing_bar_policy: none

splits:
  test_end: "2026-03-10"

pullback:
  bull_tier1_mult: 0.25
  bull_tier2_mult: 0.10
  bear_tier1_mult: 0.25
  bear_tier2_mult: 0.10
  neutral_tier1_mult: 0.15
  neutral_tier2_mult: 0.07
  max_wait_bars: 6
  runaway_atr_mult: 1.0

risk:
  atr_stop_mult: 1.5
  atr_tp_mult: 2.5
  use_signal_atr: true
  trailing_enabled: true
  trailing_atr_mult: 0.8
  break_even_atr_trigger: 0.8
  max_hold_bars: 0
  base_allocation_frac: 0.50
  init_cash: 10000.0

hmm:
  n_states: 3
  random_seed: 42
  min_prob_hard: 0.45
  min_prob_hard_short: 0.45
  min_prob_entry: 0.45
  min_prob_exit: 0.40
  min_confidence: 0.40

regime:
  persistence_bars: 3
  persistence_bars_short: 5
  transition_cooldown_bars: 0
  adx_threshold: 20.0

session:
  enabled: true
  allowed_hours_utc: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

features:
  hma_period: 55
  ema_period: 21
  atr_period: 14
  rsi_period: 14
  adx_period: 14
  dc_period: 20
  bb_period_5m: 60

smc:
  require_confluence: false
  require_pin_bar: false

smc_5m:
  enabled: true
  ob_validity_bars: 36
  fvg_validity_bars: 24
  swing_lookback_bars: 36
  ob_impulse_bars: 2
  ob_impulse_mult: 1.2
  require_confluence: false
  require_pin_bar: false

strategies:
  trend:
    enabled: true
    persistence_bars: 3
    atr_stop_mult: 1.5
    atr_tp_mult: 2.5
    trailing_enabled: true
    trailing_atr_mult: 0.8
    session_enabled: true
    allowed_hours_utc: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
  range:
    enabled: false
  volatile:
    enabled: false
  cdc:
    enabled: false
  cdc_retest:
    enabled: false

optimization:
  n_trials: 200
  target: val_sharpe
  search_space:
    bull_tier1_mult:    [0.10, 0.50]
    bull_tier2_mult:    [0.05, 0.20]
    bear_tier1_mult:    [0.10, 0.50]
    bear_tier2_mult:    [0.05, 0.20]
    neutral_tier1_mult: [0.05, 0.30]
    neutral_tier2_mult: [0.02, 0.15]
    max_wait_bars:      [3, 10]
    runaway_atr_mult:   [0.5, 2.0]
    atr_stop_mult:      [1.0, 3.0]
    atr_tp_mult:        [1.5, 4.0]
    hmm_min_prob:       [0.35, 0.65]
    trailing_atr_mult:  [0.5, 1.5]

costs:
  spread_pips: 3
  slippage_pips: 1
  commission_rt_bps: 2

pipeline:
  golden_model_threshold: 0.20
  cache_features: false
```

- [ ] **Step 2: Verify the config loads without error**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -c "
from trade2.config.loader import load_config
cfg = load_config('configs/base.yaml', 'configs/scalp_3tf.yaml')
print('entry_timeframe:', cfg['strategy']['entry_timeframe'])
print('pullback:', cfg['pullback'])
print('OK')
"
```

Expected: prints `entry_timeframe: 1M` and pullback dict, no errors.

- [ ] **Step 3: Commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/configs/scalp_3tf.yaml
git commit -m "feat: add scalp_3tf.yaml config for 3-TF pullback strategy"
```

---

## Task 2: Add `forward_fill_5m_to_1m()` in `regime.py`

**Files:**
- Modify: `code3.0/src/trade2/signals/regime.py`

This function takes the 5M signal dataframe (with signal_long, signal_short, atr_14, Close, Volume, regime columns already populated) and forward-fills all relevant columns onto the 1M bar index. It also adds `new_signal_long` and `new_signal_short` onset markers — True only on the first 1M bar corresponding to a new 5M signal.

- [ ] **Step 1: Add the function to `regime.py`**

Append to `code3.0/src/trade2/signals/regime.py`:

```python
def forward_fill_5m_to_1m(
    df_1m: pd.DataFrame,
    df_5m_signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill 5M signal columns onto 1M bars for the PullbackEngine.

    Columns forwarded from df_5m_signals:
      signal_long, signal_short, exit_long, exit_short
      stop_long, stop_short, tp_long, tp_short
      position_size_long, position_size_short
      atr_14   (forwarded as atr_5m)
      Close    (forwarded as signal_close_5m  -- 5M bar close for tier calc)
      Volume   (forwarded as vol_5m)
      regime, bull_prob, bear_prob

    Also adds:
      vol_avg_5m   -- 20-bar rolling mean of vol_5m (for vol_ratio)
      new_signal_long  -- True on first 1M bar where signal_long_5m flips 0->1
      new_signal_short -- True on first 1M bar where signal_short_5m flips 0->1

    Args:
        df_1m:          1M OHLCV DataFrame with DatetimeIndex
        df_5m_signals:  5M DataFrame after generate_signals + compute_stops

    Returns:
        df_1m copy with all forwarded columns added.
    """
    out = df_1m.copy()

    # Columns to forward-fill directly
    fwd_cols = {
        "signal_long":           "signal_long_5m",
        "signal_short":          "signal_short_5m",
        "exit_long":             "exit_long_5m",
        "exit_short":            "exit_short_5m",
        "stop_long":             "stop_long_5m",
        "stop_short":            "stop_short_5m",
        "tp_long":               "tp_long_5m",
        "tp_short":              "tp_short_5m",
        "position_size_long":    "position_size_long_5m",
        "position_size_short":   "position_size_short_5m",
        "atr_14":                "atr_5m",
        "Close":                 "signal_close_5m",
        "Volume":                "vol_5m",
        "regime":                "regime",
        "bull_prob":             "bull_prob",
        "bear_prob":             "bear_prob",
    }

    for src_col, dst_col in fwd_cols.items():
        if src_col not in df_5m_signals.columns:
            continue
        s = df_5m_signals[src_col]
        out[dst_col] = s.reindex(out.index, method="ffill").fillna(
            0 if df_5m_signals[src_col].dtype in [int, bool, "int64", "bool"] else 0.0
        )

    # 20-bar rolling mean of vol_5m (computed on 5M, then forwarded)
    if "Volume" in df_5m_signals.columns:
        vol_avg = df_5m_signals["Volume"].rolling(20, min_periods=1).mean()
        out["vol_avg_5m"] = vol_avg.reindex(out.index, method="ffill").fillna(
            df_5m_signals["Volume"].mean()
        )

    # Onset markers: True on first 1M bar where signal transitions 0->1
    for sig_col, onset_col in [
        ("signal_long_5m",  "new_signal_long"),
        ("signal_short_5m", "new_signal_short"),
    ]:
        if sig_col in out.columns:
            prev = out[sig_col].shift(1).fillna(0)
            out[onset_col] = ((out[sig_col] == 1) & (prev == 0)).astype(int)
        else:
            out[onset_col] = 0

    return out
```

- [ ] **Step 2: Verify it runs on a small slice**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -c "
import pandas as pd
from trade2.signals.regime import forward_fill_5m_to_1m

# Minimal smoke test with dummy data
idx_5m = pd.date_range('2025-01-02', periods=10, freq='5min', tz='UTC')
idx_1m = pd.date_range('2025-01-02', periods=50, freq='1min', tz='UTC')
import numpy as np
df_5m = pd.DataFrame({
    'Open': 2600.0, 'High': 2601.0, 'Low': 2599.0, 'Close': 2600.5,
    'Volume': np.random.randint(100, 500, 10).astype(float),
    'signal_long': [0,1,1,0,0,0,1,0,0,0],
    'signal_short': 0, 'exit_long': 0, 'exit_short': 0,
    'stop_long': 2598.0, 'stop_short': 2603.0,
    'tp_long': 2605.0, 'tp_short': 2595.0,
    'position_size_long': 1.0, 'position_size_short': 1.0,
    'atr_14': 2.5, 'regime': 'bull', 'bull_prob': 0.8, 'bear_prob': 0.1,
}, index=idx_5m)
df_1m = pd.DataFrame({
    'Open': 2600.0, 'High': 2601.0, 'Low': 2599.5, 'Close': 2600.5, 'Volume': 200.0,
}, index=idx_1m)
result = forward_fill_5m_to_1m(df_1m, df_5m)
print('Columns:', list(result.columns))
print('new_signal_long sum:', result['new_signal_long'].sum())
print('atr_5m sample:', result['atr_5m'].iloc[0])
print('OK')
"
```

Expected: prints column list including `new_signal_long`, `atr_5m`, `signal_close_5m`, no errors.

- [ ] **Step 3: Commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/src/trade2/signals/regime.py
git commit -m "feat: add forward_fill_5m_to_1m() for 3-TF pullback engine"
```

---

## Task 3: Create `pullback_engine.py`

**Files:**
- Create: `code3.0/src/trade2/backtesting/pullback_engine.py`

The `PullbackEngine` runs bar-by-bar on 1M OHLCV. It reads onset markers from `new_signal_long` / `new_signal_short`, computes adaptive tier limits using the 1H regime and 5M volume ratio, then manages pending limit orders and open positions.

- [ ] **Step 1: Write the engine**

Create `code3.0/src/trade2/backtesting/pullback_engine.py`:

```python
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
    atr_5m          = df_1m["atr_5m"].values.astype(float)
    vol_5m          = df_1m["vol_5m"].values.astype(float)
    vol_avg_5m      = df_1m["vol_avg_5m"].values.astype(float)
    ps_long         = df_1m["position_size_long_5m"].values.astype(float)
    ps_short        = df_1m["position_size_short_5m"].values.astype(float)
    regimes         = df_1m["regime"].values  # string array

    max_wait     = int(pb_cfg["max_wait_bars"])
    runaway_mult = float(pb_cfg["runaway_atr_mult"])
    atr_stop_mult  = float(risk_cfg["atr_stop_mult"])
    atr_tp_mult    = float(risk_cfg["atr_tp_mult"])
    trail_mult     = float(risk_cfg.get("trailing_atr_mult", 0.8))
    trail_enabled  = bool(risk_cfg.get("trailing_enabled", True))
    be_trigger     = float(risk_cfg.get("break_even_atr_trigger", 0.8))
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
    hwm       = 0.0   # high-water mark for trailing stop

    # --- Pending order state ---
    # None when inactive; dict when an order is waiting for fill
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
                    new_sl = hwm - trail_mult * atr_5m[i]
                    if new_sl > frozen_sl:
                        frozen_sl = new_sl
                else:
                    hwm = min(hwm, closes[i])
                    new_sl = hwm + trail_mult * atr_5m[i]
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

            # Check exits (priority: SL > TP > trailing > regime flip)
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
            else:  # short
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
                    "tier":          pending_tier_used if "pending_tier_used" in dir() else 0,
                })
                in_pos    = False
                direction = None
                pending   = None  # cancel any pending order on exit
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

            # 2a. Same-direction new 5M signal: refresh levels, reset counter
            if (pd_dir == "long" and new_long[i]) or (pd_dir == "short" and new_short[i]):
                t1, t2 = _compute_limits(
                    pd_dir, signal_close_5m[i], atr_5m[i],
                    vol_5m[i], vol_avg_5m[i], regimes[i], pb_cfg,
                )
                pending = {
                    "dir": pd_dir, "tier1": t1, "tier2": t2,
                    "bar_count": 0, "signal_close": signal_close_5m[i],
                    "atr": atr_5m[i], "ps": ps_long[i] if pd_dir == "long" else ps_short[i],
                }

            else:
                # 2b. Check invalidation
                atr_now = atr_5m[i]
                sc      = pending["signal_close"]
                invalidated = False

                # Regime flip (direction no longer supported)
                regime_now = str(regimes[i]).lower()
                if pd_dir == "long" and "bear" in regime_now:
                    invalidated = True
                elif pd_dir == "short" and "bull" in regime_now:
                    invalidated = True

                # Runaway move
                if not invalidated:
                    if pd_dir == "long" and closes[i] > sc + runaway_mult * atr_now:
                        invalidated = True
                    elif pd_dir == "short" and closes[i] < sc - runaway_mult * atr_now:
                        invalidated = True

                # Opposite signal
                if not invalidated:
                    if pd_dir == "long" and new_short[i]:
                        invalidated = True
                    elif pd_dir == "short" and new_long[i]:
                        invalidated = True

                if invalidated:
                    pending = None

                else:
                    # 2c. Try fills (tier1 first — better price)
                    filled = False
                    fill_px = 0.0
                    tier_hit = 0

                    if pd_dir == "long":
                        if lows[i] <= pending["tier1"]:
                            fill_px = pending["tier1"]
                            tier_hit = 1
                            filled = True
                        elif lows[i] <= pending["tier2"]:
                            fill_px = pending["tier2"]
                            tier_hit = 2
                            filled = True
                    else:  # short
                        if highs[i] >= pending["tier1"]:
                            fill_px = pending["tier1"]
                            tier_hit = 1
                            filled = True
                        elif highs[i] >= pending["tier2"]:
                            fill_px = pending["tier2"]
                            tier_hit = 2
                            filled = True

                    if filled:
                        # Apply slippage to fill
                        if pd_dir == "long":
                            fill_px = fill_px * (1.0 + slippage_arr[i])
                        else:
                            fill_px = fill_px * (1.0 - slippage_arr[i])

                        # Compute SL/TP from fill price using 5M ATR
                        fill_atr = pending["atr"]
                        if pd_dir == "long":
                            sl = fill_px - atr_stop_mult * fill_atr
                            tp = fill_px + atr_tp_mult  * fill_atr
                        else:
                            sl = fill_px + atr_stop_mult * fill_atr
                            tp = fill_px - atr_tp_mult  * fill_atr

                        pos_val  = cash * base_alloc * pending["ps"]
                        n_units  = pos_val / fill_px if fill_px > 0 else 0.0
                        in_pos    = True
                        direction = pd_dir
                        entry_bar = i
                        entry_px  = fill_px
                        frozen_sl = sl
                        frozen_tp = tp
                        entry_atr = fill_atr
                        hwm       = fill_px
                        be_active = False
                        pending_tier_used = tier_hit
                        pending   = None

                    else:
                        pending["bar_count"] += 1
                        if pending["bar_count"] >= max_wait:
                            pending = None

        # ================================================================
        # 3. Check for new 5M signal (only if no position and no pending)
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
                    signal_dir, signal_close_5m[i], atr_5m[i],
                    vol_5m[i], vol_avg_5m[i], regimes[i], pb_cfg,
                )
                pending = {
                    "dir": signal_dir, "tier1": t1, "tier2": t2,
                    "bar_count": 0, "signal_close": signal_close_5m[i],
                    "atr": atr_5m[i], "ps": ps_val,
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
            "tier":          0,
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
```

- [ ] **Step 2: Smoke test the engine with dummy data**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -c "
import numpy as np
import pandas as pd
from trade2.backtesting.pullback_engine import simulate

idx = pd.date_range('2025-01-02', periods=200, freq='1min', tz='UTC')
np.random.seed(42)
prices = 2600 + np.cumsum(np.random.randn(200) * 0.3)
df = pd.DataFrame({
    'Open': prices, 'High': prices + 0.5, 'Low': prices - 0.5, 'Close': prices,
    'Volume': 200.0,
    'new_signal_long':  ([0]*10 + [1] + [0]*9) * 10,
    'new_signal_short': 0,
    'signal_long_5m':   ([0]*10 + [1]*10) * 10,
    'signal_short_5m':  0,
    'exit_long_5m': 0, 'exit_short_5m': 0,
    'signal_close_5m': prices,
    'atr_5m': 2.5, 'vol_5m': 300.0, 'vol_avg_5m': 250.0,
    'position_size_long_5m': 1.0, 'position_size_short_5m': 1.0,
    'regime': 'bull', 'bull_prob': 0.8, 'bear_prob': 0.1,
}, index=idx)

pb_cfg = {
    'bull_tier1_mult': 0.25, 'bull_tier2_mult': 0.10,
    'bear_tier1_mult': 0.25, 'bear_tier2_mult': 0.10,
    'neutral_tier1_mult': 0.15, 'neutral_tier2_mult': 0.07,
    'max_wait_bars': 6, 'runaway_atr_mult': 1.0,
}
risk_cfg = {
    'atr_stop_mult': 1.5, 'atr_tp_mult': 2.5, 'trailing_atr_mult': 0.8,
    'trailing_enabled': True, 'break_even_atr_trigger': 0.8,
    'base_allocation_frac': 0.5,
}
equity, trades = simulate(df, pb_cfg, risk_cfg, 10000.0, 0.0002, np.zeros(200))
print('Equity final:', round(equity.iloc[-1], 2))
print('Trades:', len(trades))
if len(trades) > 0:
    print(trades[['direction','entry_price','exit_price','pnl','tier','exit_reason']].head())
print('OK')
"
```

Expected: runs without error, prints trade count >= 0, no exceptions.

- [ ] **Step 3: Commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/src/trade2/backtesting/pullback_engine.py
git commit -m "feat: add PullbackEngine for 3-TF tiered limit order simulation"
```

---

## Task 4: Wire 3-TF Data Loading in `run_pipeline.py`

**Files:**
- Modify: `code3.0/src/trade2/app/run_pipeline.py`

When `config['strategy']['entry_timeframe']` is set to `"1M"`, the pipeline:
1. Loads 1M data splits
2. After signal generation on 5M, calls `forward_fill_5m_to_1m()`
3. Runs `pullback_engine.simulate()` instead of `run_backtest()`
4. Saves trades CSV and metrics JSON using the same naming convention

- [ ] **Step 1: Add imports at top of `run_pipeline.py`**

After the existing imports block (around line 40), add:

```python
from trade2.signals.regime import forward_fill_5m_to_1m
from trade2.backtesting import pullback_engine
from trade2.backtesting.costs import compute_slippage_array
```

- [ ] **Step 2: Add `_run_3tf_split()` helper function**

Add this function after `_log_signal_stats()` (around line 156) in `run_pipeline.py`:

```python
def _run_3tf_split(
    df_1m: pd.DataFrame,
    df_5m_signals: pd.DataFrame,
    config: dict,
    period_label: str,
    strategy_name: str,
    backtests_dir: Path,
) -> dict:
    """
    Run one split (train/val/test) of the 3-TF pullback backtest.

    Args:
        df_1m:          1M OHLCV split (already date-sliced)
        df_5m_signals:  5M df after generate_signals + compute_stops (same period)
        config:         full config dict
        period_label:   'train', 'val', or 'test'
        strategy_name:  used for output file naming
        backtests_dir:  where to save CSV / JSON

    Returns:
        metrics dict (same schema as run_backtest)
    """
    from trade2.backtesting.metrics import compute_metrics, format_report

    risk_cfg = config["risk"]
    costs    = config["costs"]
    pb_cfg   = config["pullback"]

    # Forward-fill 5M signals onto 1M bars
    df_1m_ready = forward_fill_5m_to_1m(df_1m, df_5m_signals)

    # Slippage array (1M bars)
    slippage_arr = compute_slippage_array(
        df_1m_ready,
        slippage_pips=costs["slippage_pips"],
        pip_size=0.01,
    )
    commission_rt = costs["commission_rt_bps"] / 10000.0 * 2  # round-trip

    init_cash = risk_cfg["init_cash"]

    equity, trades_df = pullback_engine.simulate(
        df_1m_ready, pb_cfg, risk_cfg, init_cash, commission_rt, slippage_arr,
    )

    # Compute metrics (signal_tf=1M -> bars_per_year = 252*390 = ~98280)
    metrics = compute_metrics(equity, trades_df, bars_per_year=98280)

    # Persist trades CSV
    if backtests_dir is not None:
        trades_path = backtests_dir / f"{strategy_name}_{period_label}_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"  [3tf] {period_label}: {len(trades_df)} trades saved -> {trades_path.name}")

    # Persist metrics JSON
    if backtests_dir is not None:
        metrics_path = backtests_dir / f"{strategy_name}_{period_label}.json"
        import json
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    _print_split_metrics(period_label, metrics)
    return metrics


def _print_split_metrics(label: str, m: dict) -> None:
    print(f"  [{label}] return={m.get('annualized_return',0)*100:.1f}% "
          f"sharpe={m.get('sharpe_ratio',0):.2f} "
          f"dd={m.get('max_drawdown',0)*100:.1f}% "
          f"trades={m.get('total_trades',0)} "
          f"wr={m.get('win_rate',0)*100:.1f}%")
```

- [ ] **Step 3: Add 3-TF path inside `run_pipeline()`**

In `run_pipeline.py`, inside `run_pipeline()`, find the block that starts `if mode == "multi_tf":` and loads signal TF data (around line 269). After the signal generation + compute_stops block for train/val/test, add:

```python
    entry_tf = config.get("strategy", {}).get("entry_timeframe", None)

    if entry_tf == "1M" and mode == "multi_tf":
        print(f"\n[pipeline] 3-TF mode: loading 1M entry bars...")
        train_1m, val_1m, test_1m = load_split_tf("1M", config)
        print(f"  1M train={len(train_1m)} | val={len(val_1m)} | test={len(test_1m)}")

        print(f"[pipeline] Running 3-TF pullback backtests...")
        train_metrics = _run_3tf_split(train_1m, train_sig_final, config, "train", strategy_name, dirs["backtests"])
        val_metrics   = _run_3tf_split(val_1m,   val_sig_final,   config, "val",   strategy_name, dirs["backtests"])
        test_metrics  = _run_3tf_split(test_1m,  test_sig_final,  config, "test",  strategy_name, dirs["backtests"])

        # Use val_metrics for optimization target; return early with simple result
        result = {
            "train": train_metrics,
            "val":   val_metrics,
            "test":  test_metrics,
            "strategy_name": strategy_name,
        }
        print(f"\n[pipeline] 3-TF run complete.")
        return result
```

Note: `train_sig_final`, `val_sig_final`, `test_sig_final` are the variables that hold the 5M signal dataframes after `generate_signals` + `compute_stops` + `route_signals`. Check the actual variable names in the existing code at that point (they are `train_sig`, `val_sig`, `test_sig` after being processed through the signal chain). Use the correct names as found in the file.

- [ ] **Step 4: Verify the 3-TF pipeline runs end-to-end on a dry run**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -c "
from trade2.config.loader import load_config
from trade2.app.run_pipeline import run_pipeline
cfg = load_config('configs/base.yaml', 'configs/scalp_3tf.yaml')
# Use a narrow date range to test quickly -- override splits
cfg['splits']['train_end'] = '2020-06-30'
cfg['splits']['val_end']   = '2020-12-31'
cfg['splits']['test_end']  = '2021-03-31'
result = run_pipeline(cfg, retrain_model=True, walk_forward=False)
print('train return:', result['train'].get('annualized_return'))
print('val trades:',   result['val'].get('total_trades'))
print('test trades:',  result['test'].get('total_trades'))
print('OK')
"
```

Expected: completes without error, prints metrics for all three splits.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/src/trade2/app/run_pipeline.py
git commit -m "feat: wire 3-TF 1M pullback path into run_pipeline"
```

---

## Task 5: Add Pullback Params to Optimizer

**Files:**
- Modify: `code3.0/src/trade2/optimization/optimizer.py`

Add a `run_optimization_3tf()` function that pre-computes the 5M signals once, then iterates Optuna trials by varying pullback params and re-running `pullback_engine.simulate()` on the 1M val bars.

- [ ] **Step 1: Add imports at top of `optimizer.py`**

After the existing imports, add:

```python
import copy
from trade2.signals.regime import forward_fill_5m_to_1m
from trade2.backtesting import pullback_engine
```

- [ ] **Step 2: Add `run_optimization_3tf()` to `optimizer.py`**

Append to `code3.0/src/trade2/optimization/optimizer.py`:

```python
def run_optimization_3tf(
    val_5m_signals: pd.DataFrame,   # val 5M df with signals + stops already computed
    val_1m: pd.DataFrame,           # val 1M OHLCV (raw, not yet forward-filled)
    config: Dict[str, Any],
    n_trials: int = 200,
    optuna_target: str = "val_sharpe",
) -> Dict[str, Any]:
    """
    Optuna optimization for the 3-TF pullback strategy.

    Pre-computes forward-filled 1M base (without pullback params) once.
    Each trial varies pullback + risk params, re-runs simulate(), returns target metric.

    Returns:
        dict with keys: best_params, best_value, study
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    costs    = config["costs"]
    risk_cfg = config["risk"]
    ss       = config.get("optimization", {}).get("search_space", {})

    # Pre-compute slippage array once (doesn't change between trials)
    slippage_arr = compute_slippage_array(
        val_1m,
        slippage_pips=costs["slippage_pips"],
        pip_size=0.01,
    )
    commission_rt = costs["commission_rt_bps"] / 10000.0 * 2
    init_cash     = risk_cfg["init_cash"]

    # Pre-forward-fill 5M signals onto 1M (base columns only, no pullback-specific values)
    val_1m_base = forward_fill_5m_to_1m(val_1m, val_5m_signals)

    def objective(trial: "optuna.Trial") -> float:
        # Sample pullback params
        pb_trial = copy.deepcopy(config["pullback"])

        for key, bounds in ss.items():
            if key in ("bull_tier1_mult", "bull_tier2_mult",
                       "bear_tier1_mult", "bear_tier2_mult",
                       "neutral_tier1_mult", "neutral_tier2_mult",
                       "runaway_atr_mult", "atr_stop_mult", "atr_tp_mult",
                       "hmm_min_prob", "trailing_atr_mult"):
                val = trial.suggest_float(key, bounds[0], bounds[1])
                if key in pb_trial:
                    pb_trial[key] = val
            elif key == "max_wait_bars":
                pb_trial[key] = trial.suggest_int(key, int(bounds[0]), int(bounds[1]))

        # Enforce tier2 < tier1 for each regime
        for regime_prefix in ("bull_", "bear_", "neutral_"):
            t1_key = f"{regime_prefix}tier1_mult"
            t2_key = f"{regime_prefix}tier2_mult"
            if pb_trial[t2_key] >= pb_trial[t1_key]:
                pb_trial[t2_key] = pb_trial[t1_key] * 0.5

        # Risk params from trial
        risk_trial = copy.deepcopy(risk_cfg)
        if "atr_stop_mult" in ss:
            risk_trial["atr_stop_mult"] = trial.suggest_float("atr_stop_mult", ss["atr_stop_mult"][0], ss["atr_stop_mult"][1])
        if "atr_tp_mult" in ss:
            risk_trial["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", ss["atr_tp_mult"][0], ss["atr_tp_mult"][1])
        if "trailing_atr_mult" in ss:
            risk_trial["trailing_atr_mult"] = trial.suggest_float("trailing_atr_mult", ss["trailing_atr_mult"][0], ss["trailing_atr_mult"][1])

        try:
            equity, trades_df = pullback_engine.simulate(
                val_1m_base, pb_trial, risk_trial, init_cash, commission_rt, slippage_arr,
            )
            if len(trades_df) < 10:
                return -999.0
            metrics = compute_metrics(equity, trades_df, bars_per_year=98280)
            target_map = {
                "val_sharpe":  metrics.get("sharpe_ratio", -999),
                "val_return":  metrics.get("annualized_return", -999),
                "val_calmar":  metrics.get("calmar_ratio", -999),
            }
            return float(target_map.get(optuna_target, metrics.get("sharpe_ratio", -999)))
        except Exception:
            return -999.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[optuna-3tf] Best {optuna_target}: {study.best_value:.4f}")
    print(f"[optuna-3tf] Best params: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value":  study.best_value,
        "study":       study,
    }
```

- [ ] **Step 3: Verify import works**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -c "from trade2.optimization.optimizer import run_optimization_3tf; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/src/trade2/optimization/optimizer.py
git commit -m "feat: add run_optimization_3tf() for pullback param Optuna search"
```

---

## Task 6: Full Pipeline Smoke Test + Optimization Run

**Files:** (none created — verification only)

- [ ] **Step 1: Run full train/val/test pipeline with default params**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
trade2 --config configs/base.yaml --config configs/scalp_3tf.yaml --retrain-model --skip-walk-forward
```

Expected output (approximate):
```
[pipeline] 3-TF mode: loading 1M entry bars...
[pipeline] Running 3-TF pullback backtests...
  [train] return=X% sharpe=X.XX dd=-X.X% trades=NNN wr=XX.X%
  [val]   return=X% sharpe=X.XX dd=-X.X% trades=NNN wr=XX.X%
  [test]  return=X% sharpe=X.XX dd=-X.X% trades=NNN wr=XX.X%
[pipeline] 3-TF run complete.
```

Check: `trades` > 0 in all splits, no Python exceptions.

- [ ] **Step 2: Check fill distribution in train trades**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -c "
import pandas as pd
df = pd.read_csv('artefacts/backtests/xauusd_scalp_3tf_train_trades.csv')
print('Total trades:', len(df))
print('Tier distribution:')
print(df['tier'].value_counts())
print('Instant exits (0 bars):', (df['duration_bars']==0).sum())
print('Exit reasons:')
print(df['exit_reason'].value_counts())
print('Win rate:', round((df['pnl']>0).mean()*100, 1), '%')
"
```

Expected: tier 1 and tier 2 both appear (not all on one tier), instant exits near 0%.

- [ ] **Step 3: Run Optuna optimization (50 trials quick test)**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
trade2 --config configs/base.yaml --config configs/scalp_3tf.yaml --optimize --trials 50 --skip-walk-forward
```

Expected: runs 50 trials, prints best Sharpe and params. No errors.

- [ ] **Step 4: Run full optimization (200 trials)**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
trade2 --config configs/base.yaml --config configs/scalp_3tf.yaml --optimize --trials 200 --skip-walk-forward
```

Expected: completes, prints best params. Note the best params for manual config update.

- [ ] **Step 5: Update `scalp_3tf.yaml` with best Optuna params, re-run test**

After Step 4, manually update `pullback`, `risk` sections in `scalp_3tf.yaml` with the best Optuna params. Then run:

```bash
trade2 --config configs/base.yaml --config configs/scalp_3tf.yaml --skip-walk-forward
```

Expected test metrics: win rate > 45%, profit factor > 1.2, Sharpe > 1.0 (first pass).

- [ ] **Step 6: Final commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/configs/scalp_3tf.yaml
git commit -m "feat: update scalp_3tf with optimized pullback params from Optuna"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] 1H regime / 5M signal / 1M entry architecture — Task 4
- [x] Adaptive tiered limits (regime + volume) — Task 3 `_compute_limits()`
- [x] Both tiers active from bar 1 — Task 3 step 2c
- [x] Tier1 checked before tier2 — Task 3 explicit order
- [x] Fill price = limit price (not bar low/high) — Task 3 `fill_px = pending["tier1"]`
- [x] Invalidation: regime flip, runaway, opposite signal — Task 3 step 2b
- [x] Same-direction refresh — Task 3 step 2a
- [x] Cancel after `max_wait_bars` — Task 3 step 2c
- [x] SL/TP from fill price + 5M ATR — Task 3
- [x] Trailing stop — Task 3
- [x] Break-even stop — Task 3
- [x] No timeout (`max_hold_bars: 0`) — config
- [x] Optuna search space for all pullback params — Task 5
- [x] tier2_mult < tier1_mult constraint enforced — Task 5 `objective()`
- [x] Fill distribution diagnostic — Task 6 Step 2

**Type consistency:**
- `pullback_engine.simulate()` signature matches calls in `_run_3tf_split()` and `run_optimization_3tf()` ✓
- `forward_fill_5m_to_1m()` output columns match what `pullback_engine.simulate()` reads ✓
- `_run_3tf_split()` uses `compute_metrics(bars_per_year=98280)` — 252 trading days × 390 minutes/day ✓
