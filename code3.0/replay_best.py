"""
replay_best.py
Exact replay of experiment 20260315_074921 (idea_16 4H 2-state, 43.6% return, 89 trades).

Key facts about the original run:
  - full_scheme_search used legacy_signals=True (NOT regime_specialized)
  - config = base.yaml + xauusd_mtf.yaml + idea_16 override (regime_timeframe=4H, n_states=2)
  - optimizer found best params, applied via flat p[] dict
  - NO drawdown_filter, NO freshness filter (added later in best_4h_43pct.yaml)
  - max_hold_bars=48 (before the "no timeout" rule was enforced)
  - HMM retrained on train split (4H, 2 states)

Outputs:
  artefacts/best_trades.csv   -- full trade log
  artefacts/best_chart.html   -- interactive Plotly chart
"""

import sys, json
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trade2.config.loader import load_config
from trade2.app.run_pipeline import run_pipeline

# -----------------------------------------------------------------------
# 1. Build exact original config
# -----------------------------------------------------------------------
BASE  = ROOT / "configs" / "base.yaml"
OVER  = ROOT / "configs" / "xauusd_mtf.yaml"

config = load_config(BASE, OVER)

# idea_16 override (exactly as in full_scheme_search.py)
def _deep_merge(base, override):
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result

idea16_override = {
    "strategy": {"regime_timeframe": "4H"},
    "hmm":      {"n_states": 2, "sizing_max": 2.0},
    "risk":     {"base_allocation_frac": 0.90},
}
config = _deep_merge(config, idea16_override)

# Remove any filters added after this experiment
config.pop("drawdown_filter",  None)
config.get("regime", {}).pop("max_regime_freshness",   None)
config.get("regime", {}).pop("freshness_decay_start",  None)

# Restore original max_hold_bars (before the no-timeout rule)
config["risk"]["max_hold_bars"] = 48

# Restore original experiment splits (splits.py uses train_end + val_end only)
# Original: train=2019-2022, val=2023, test=2024-onwards
config["splits"]["train_end"] = "2022-12-31"
config["splits"]["val_end"]   = "2023-12-31"

# Turn off feature caching
config.setdefault("pipeline", {})["cache_features"] = False

# -----------------------------------------------------------------------
# 2. Exact optimized params from experiment 20260315_074921
# -----------------------------------------------------------------------
BEST_PARAMS = {
    "hma_period":              55,
    "ema_period":              21,
    "atr_period":              14,
    "rsi_period":              14,
    "adx_period":              14,
    "dc_period":               40,
    "adx_threshold":           16.715103754797116,
    "hmm_min_prob":            0.7348404387333658,
    "hmm_states":              2,
    "regime_persistence_bars": 4,
    "atr_stop_mult":           3.095514338084446,
    "atr_tp_mult":             4.488045570804698,
    "require_smc_confluence":  True,
    "require_pin_bar":         True,
    "hmm_min_confidence":      0.5117980796648711,
    "transition_cooldown_bars": 1,
}

# Also patch into config so generate_signals reads min_confidence / cooldown correctly
config["hmm"]["min_confidence"]             = BEST_PARAMS["hmm_min_confidence"]
config["regime"]["transition_cooldown_bars"] = BEST_PARAMS["transition_cooldown_bars"]
config["regime"]["persistence_bars"]         = BEST_PARAMS["regime_persistence_bars"]
config["regime"]["adx_threshold"]            = BEST_PARAMS["adx_threshold"]

# -----------------------------------------------------------------------
# 3. Run pipeline with legacy_signals=True (same as original full_scheme_search)
# -----------------------------------------------------------------------
print("[replay] Reproducing experiment 20260315_074921")
print("[replay] 4H 2-state HMM | legacy signals | test 2024-01-01 to 2025-06-30")
print()

results = run_pipeline(
    config          = config,
    params          = BEST_PARAMS,
    walk_forward    = False,
    retrain_model   = False,
    export_approved = False,
    legacy_signals  = True,
    return_trades   = True,
)

test_metrics = results.get("test_metrics", {})
trades_raw   = results.get("test_trades", [])

print(f"\n[replay] Test results:")
print(f"  Return : {test_metrics.get('annualized_return', 0)*100:.1f}%")
print(f"  Sharpe : {test_metrics.get('sharpe_ratio', 0):.3f}")
print(f"  Max DD : {test_metrics.get('max_drawdown', 0)*100:.1f}%")
print(f"  Trades : {test_metrics.get('total_trades', 0)}")
print(f"  Win %  : {test_metrics.get('win_rate', 0)*100:.1f}%")
print(f"  PF     : {test_metrics.get('profit_factor', 0):.3f}")

if not trades_raw:
    print("[replay] No trades returned.")
    sys.exit(1)

trades = pd.DataFrame(trades_raw)
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
trades["win"]        = trades["pnl"] > 0

# Save CSV
art_dir = ROOT / "artefacts"
art_dir.mkdir(exist_ok=True)
csv_path = art_dir / "best_trades.csv"
trades.to_csv(csv_path, index=False)
print(f"\n[replay] Trade log saved: {csv_path}")

# Print trade table
print("\n" + "="*100)
print("TRADE LOG")
print("="*100)
cols = ["entry_time", "exit_time", "direction", "entry_price", "exit_price", "sl", "tp", "pnl", "exit_reason"]
print(trades[cols].to_string(index=False))
print("="*100)
wins   = trades["win"].sum()
losses = (~trades["win"]).sum()
print(f"Trades: {len(trades)}  Wins: {wins}  Losses: {losses}  Total PnL: ${trades['pnl'].sum():.0f}")

# -----------------------------------------------------------------------
# 4. Build Plotly chart
# -----------------------------------------------------------------------
DATA_ROOT = ROOT.parent
raw_1h = DATA_ROOT / "data" / "raw" / "XAUUSD_1H_2019_2025.csv"

price = pd.read_csv(raw_1h, parse_dates=["datetime"])
price = price.set_index("datetime").sort_index()
price.index = price.index.tz_localize("UTC") if price.index.tzinfo is None else price.index.tz_convert("UTC")
price.columns = [c.capitalize() for c in price.columns]
price = price[(price.index >= "2024-01-01") & (price.index <= "2025-07-31")]

# Align entry/exit times to UTC
trades["entry_time"] = trades["entry_time"].dt.tz_convert("UTC") if trades["entry_time"].dt.tz else trades["entry_time"].dt.tz_localize("UTC")
trades["exit_time"]  = trades["exit_time"].dt.tz_convert("UTC")  if trades["exit_time"].dt.tz  else trades["exit_time"].dt.tz_localize("UTC")

# Equity curve: running cumulative PnL at each bar
init_cash = config["backtest"]["init_cash"]
trade_sorted = trades.sort_values("exit_time")
eq_vals = []
for t in price.index:
    closed_pnl = trade_sorted.loc[trade_sorted["exit_time"] <= t, "pnl"].sum()
    eq_vals.append(init_cash + closed_pnl)
equity = pd.Series(eq_vals, index=price.index)

COL_BULL = "#26a69a"; COL_BEAR = "#ef5350"
COL_LONG = "#00bcd4"; COL_SHORT = "#ff9800"
COL_SL   = "#f44336"; COL_TP   = "#4caf50"
COL_EQ   = "#7c4dff"

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.72, 0.28], vertical_spacing=0.04,
    subplot_titles=[
        "XAUUSD (1H) | Best Strategy: 4H 2-State HMM | Test 2024-2025",
        "Equity Curve (USD)"
    ]
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=price.index, open=price["Open"], high=price["High"],
    low=price["Low"], close=price["Close"],
    name="XAUUSD",
    increasing_line_color=COL_BULL, decreasing_line_color=COL_BEAR,
    increasing_fillcolor=COL_BULL,  decreasing_fillcolor=COL_BEAR,
    line=dict(width=1),
), row=1, col=1)

# Trade shading + SL/TP lines
for _, t in trades.iterrows():
    clr = COL_LONG if t["direction"] == "long" else COL_SHORT
    fig.add_vrect(x0=t["entry_time"], x1=t["exit_time"],
                  fillcolor=clr, opacity=0.05, line_width=0,
                  row=1, col=1, layer="below")
    fig.add_shape(type="line", x0=t["entry_time"], x1=t["exit_time"],
                  y0=t["sl"], y1=t["sl"],
                  line=dict(color=COL_SL, width=1, dash="dot"), row=1, col=1)
    fig.add_shape(type="line", x0=t["entry_time"], x1=t["exit_time"],
                  y0=t["tp"], y1=t["tp"],
                  line=dict(color=COL_TP, width=1, dash="dot"), row=1, col=1)

# Entry markers
longs  = trades[trades["direction"] == "long"]
shorts = trades[trades["direction"] == "short"]

fig.add_trace(go.Scatter(
    x=longs["entry_time"], y=longs["entry_price"], mode="markers",
    marker=dict(symbol="triangle-up", size=14, color=COL_LONG, line=dict(color="white", width=1)),
    name="Long Entry",
    hovertemplate="<b>LONG ENTRY</b><br>%{x}<br>$%{y:.2f}<extra></extra>",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=shorts["entry_time"], y=shorts["entry_price"], mode="markers",
    marker=dict(symbol="triangle-down", size=14, color=COL_SHORT, line=dict(color="white", width=1)),
    name="Short Entry",
    hovertemplate="<b>SHORT ENTRY</b><br>%{x}<br>$%{y:.2f}<extra></extra>",
), row=1, col=1)

# Exit markers: shape = exit reason, color = win/loss
SYM = {"sl": "square", "tp": "diamond", "signal": "circle", "timeout": "x", "end_of_data": "cross"}
for _, t in trades.iterrows():
    clr  = "#4caf50" if t["pnl"] > 0 else "#f44336"
    sym  = SYM.get(t["exit_reason"], "circle")
    sign = "+" if t["pnl"] > 0 else ""
    fig.add_trace(go.Scatter(
        x=[t["exit_time"]], y=[t["exit_price"]], mode="markers",
        marker=dict(symbol=sym, size=11, color=clr, line=dict(color="white", width=1)),
        showlegend=False,
        hovertemplate=(
            f"<b>EXIT {t['exit_reason'].upper()}</b><br>"
            f"Dir: {t['direction']}<br>"
            f"Entry: ${t['entry_price']:.2f}<br>"
            f"Exit: ${t['exit_price']:.2f}<br>"
            f"SL: ${t['sl']:.2f}  TP: ${t['tp']:.2f}<br>"
            f"PnL: {sign}{t['pnl']:.0f} USD<br>"
            f"Duration: {t['duration_bars']} bars<extra></extra>"
        ),
    ), row=1, col=1)

# Equity curve
fig.add_trace(go.Scatter(
    x=equity.index, y=equity.values, mode="lines",
    line=dict(color=COL_EQ, width=2), name="Equity",
    fill="tozeroy", fillcolor="rgba(124,77,255,0.08)",
), row=2, col=1)
fig.add_hline(y=init_cash, line=dict(color="gray", width=1, dash="dash"), row=2, col=1)

total_ret = (equity.iloc[-1] - init_cash) / init_cash * 100
fig.update_layout(
    template="plotly_dark", height=940,
    title=dict(
        text=(
            f"<b>XAUUSD Best Strategy (Exact Replay)</b>  |  "
            f"Ann. Return: {test_metrics.get('annualized_return',0)*100:.1f}%  "
            f"Sharpe: {test_metrics.get('sharpe_ratio',0):.2f}  "
            f"MaxDD: {test_metrics.get('max_drawdown',0)*100:.1f}%  "
            f"PF: {test_metrics.get('profit_factor',0):.2f}  "
            f"Trades: {len(trades)}  "
            f"Wins: {wins}  Losses: {losses}"
        ),
        font=dict(size=13),
    ),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=60, r=40, t=80, b=40),
    hovermode="x unified",
)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1, gridcolor="#2d2d2d")
fig.update_yaxes(title_text="Equity (USD)", row=2, col=1, gridcolor="#2d2d2d")
fig.update_xaxes(gridcolor="#2d2d2d", showgrid=True)

html_path = art_dir / "best_chart.html"
fig.write_html(str(html_path), include_plotlyjs="cdn")
print(f"\n[replay] Chart saved: {html_path.resolve()}")
print("[replay] Open the HTML file in your browser.")
