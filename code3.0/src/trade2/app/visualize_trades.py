"""
visualize_trades.py - Interactive Plotly dashboard for the +43.9% trade log.

Usage:
    python src/trade2/app/visualize_trades.py
    python src/trade2/app/visualize_trades.py --html artefacts/best_4h_43pct_dashboard.html
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── helpers ────────────────────────────────────────────────────────────────

def _load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["trade_num"]  = range(1, len(df) + 1)
    df["win"]        = df["pnl"] > 0
    df["month"]      = df["exit_time"].dt.to_period("M").dt.to_timestamp()
    df["hold_hours"] = df["duration_bars"] * 5 / 60
    return df


def _load_price(config_override: Path | None = None) -> pd.DataFrame:
    """Load 5M test bars using trade2 data pipeline."""
    try:
        from trade2.config.loader import load_config
        from trade2.data.splits import load_split_tf

        base = PROJECT_ROOT / "configs" / "base.yaml"
        override = config_override or (PROJECT_ROOT / "configs" / "best_4h_43pct.yaml")
        cfg = load_config(str(base), str(override))
        _, _, test = load_split_tf("5M", cfg)
        # Downsample to 1H for chart readability
        price = test["Close"].resample("1h").last().dropna()
        price.name = "close"
        return price.reset_index().rename(columns={"Datetime": "time", "index": "time"})
    except Exception as e:
        print(f"  [warn] Could not load price data: {e}")
        return pd.DataFrame()


def _equity_curve(trades: pd.DataFrame, init_cash: float = 100_000) -> pd.Series:
    curve = pd.Series(
        [init_cash] + list(init_cash + trades["cumulative_pnl"]),
        index=[trades["entry_time"].iloc[0]] + list(trades["exit_time"]),
    )
    return curve


# ── figure builders ────────────────────────────────────────────────────────

def _fig_price_with_trades(trades: pd.DataFrame, price_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("XAUUSD 1H — Entry / Exit Markers", "Cumulative PnL ($)"),
    )

    # --- price line ---
    if not price_df.empty:
        time_col = "time" if "time" in price_df.columns else price_df.columns[0]
        fig.add_trace(go.Scatter(
            x=price_df[time_col], y=price_df["close"],
            mode="lines", name="XAUUSD",
            line=dict(color="#888", width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Price: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # --- trade markers (split by direction & win/loss) ---
    for direction, win, color, symbol, label in [
        ("long",  True,  "#00cc44", "triangle-up",        "Long Win"),
        ("long",  False, "#ff4444", "triangle-up-open",   "Long Loss"),
        ("short", True,  "#0099ff", "triangle-down",      "Short Win"),
        ("short", False, "#ff9900", "triangle-down-open", "Short Loss"),
    ]:
        mask = (trades["direction"] == direction) & (trades["win"] == win)
        sub  = trades[mask]
        if sub.empty:
            continue
        hover = (
            sub.apply(lambda r:
                f"#{r.trade_num}  {r.direction.upper()}  {r.exit_reason}<br>"
                f"Entry: {r.entry_price:.2f}  Exit: {r.exit_price:.2f}<br>"
                f"PnL: ${r.pnl:+,.0f}   Hold: {r.hold_hours:.1f}h", axis=1)
        )
        fig.add_trace(go.Scatter(
            x=sub["entry_time"], y=sub["entry_price"],
            mode="markers", name=label,
            marker=dict(symbol=symbol, size=10, color=color,
                        line=dict(width=1, color="white")),
            text=hover, hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # --- SL / TP lines for each trade (thin, low opacity) ---
    for _, r in trades.iterrows():
        t0, t1 = r["entry_time"], r["exit_time"]
        clr = "#00cc44" if r["direction"] == "long" else "#0099ff"
        fig.add_shape(type="line", x0=t0, x1=t1, y0=r["sl"], y1=r["sl"],
                      line=dict(color="#ff4444", width=0.6, dash="dot"),
                      row=1, col=1)
        if r["tp"] > 0:
            fig.add_shape(type="line", x0=t0, x1=t1, y0=r["tp"], y1=r["tp"],
                          line=dict(color=clr, width=0.6, dash="dot"),
                          row=1, col=1)

    # --- equity curve ---
    equity = _equity_curve(trades)
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Equity",
        fill="tozeroy",
        line=dict(color="#00cc44", width=2),
        fillcolor="rgba(0,204,68,0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>Equity: $%{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        title="Best 4H 2-State HMM Strategy  |  Test 2024-01 to 2025-06  |  +43.9% / Sharpe 1.87",
        height=750,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)",  row=2, col=1)
    fig.update_xaxes(title_text="Date",        row=2, col=1)
    return fig


def _fig_monthly_pnl(trades: pd.DataFrame) -> go.Figure:
    monthly = trades.groupby("month").agg(
        pnl=("pnl", "sum"),
        trades=("pnl", "count"),
        wins=("win", "sum"),
    ).reset_index()
    monthly["label"] = monthly["month"].dt.strftime("%b %Y")
    monthly["color"] = monthly["pnl"].apply(lambda x: "#00cc44" if x >= 0 else "#ff4444")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["label"], y=monthly["pnl"],
        marker_color=monthly["color"],
        text=monthly.apply(lambda r: f"${r.pnl:+,.0f}<br>{int(r.wins)}/{int(r.trades)}", axis=1),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>PnL: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="white", width=1, dash="dot"))
    fig.update_layout(
        title="Monthly PnL  (wins / total trades shown on bars)",
        height=400, template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        yaxis_title="PnL ($)", xaxis_title="Month",
    )
    return fig


def _fig_trade_analysis(trades: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "PnL per Trade (#)",
            "Hold Duration Distribution",
            "Exit Reason Breakdown",
            "Win/Loss by Direction",
        ),
        specs=[
            [{"type": "xy"},   {"type": "xy"}],
            [{"type": "pie"},  {"type": "xy"}],
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # 1. PnL waterfall per trade
    colors = ["#00cc44" if p > 0 else "#ff4444" for p in trades["pnl"]]
    fig.add_trace(go.Bar(
        x=trades["trade_num"], y=trades["pnl"],
        marker_color=colors, name="Trade PnL",
        hovertemplate="Trade #%{x}<br>PnL: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # 2. Hold duration histogram
    fig.add_trace(go.Histogram(
        x=trades["hold_hours"], nbinsx=20,
        marker_color="#0099ff", name="Hold Hours",
        hovertemplate="Hold: %{x:.0f}h<br>Count: %{y}<extra></extra>",
    ), row=1, col=2)

    # 3. Exit reason pie
    reason_counts = trades["exit_reason"].value_counts()
    exit_colors = {
        "tp": "#00cc44", "sl": "#ff4444",
        "timeout": "#888", "signal": "#0099ff", "end_of_data": "#ff9900",
    }
    fig.add_trace(go.Pie(
        labels=reason_counts.index,
        values=reason_counts.values,
        marker_colors=[exit_colors.get(r, "#ccc") for r in reason_counts.index],
        name="Exit Reasons",
        hole=0.35,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} trades (%{percent})<extra></extra>",
    ), row=2, col=1)

    # 4. Win/loss by direction grouped bar
    for direction, color_w, color_l in [("long", "#00cc44", "#ff6666"), ("short", "#0088ff", "#ff9900")]:
        sub = trades[trades["direction"] == direction]
        wins   = int((sub["pnl"] > 0).sum())
        losses = int((sub["pnl"] <= 0).sum())
        fig.add_trace(go.Bar(
            x=[direction], y=[wins],
            name=f"{direction} Win", marker_color=color_w,
            hovertemplate=f"{direction} Wins: {wins}<extra></extra>",
        ), row=2, col=2)
        fig.add_trace(go.Bar(
            x=[direction], y=[losses],
            name=f"{direction} Loss", marker_color=color_l,
            hovertemplate=f"{direction} Losses: {losses}<extra></extra>",
        ), row=2, col=2)

    fig.update_layout(
        title="Trade Analysis",
        height=650, template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        barmode="stack", showlegend=False,
    )
    fig.update_xaxes(title_text="Trade #",        row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)",         row=1, col=1)
    fig.update_xaxes(title_text="Hold (hours)",    row=1, col=2)
    fig.update_yaxes(title_text="# Trades",        row=1, col=2)
    fig.update_yaxes(title_text="# Trades",        row=2, col=2)
    return fig


def _fig_trade_table(trades: pd.DataFrame) -> go.Figure:
    disp = trades.copy()
    disp["entry"] = disp["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    disp["exit"]  = disp["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
    disp["pnl_fmt"]   = disp["pnl"].apply(lambda x: f"${x:+,.0f}")
    disp["hold_fmt"]  = disp["hold_hours"].apply(lambda x: f"{x:.1f}h")
    disp["entry_fmt"] = disp["entry_price"].apply(lambda x: f"{x:.2f}")
    disp["exit_fmt"]  = disp["exit_price"].apply(lambda x: f"{x:.2f}")
    disp["result"]    = disp["win"].map({True: "WIN", False: "LOSS"})

    cell_colors = []
    # pnl column color
    pnl_colors = ["rgba(0,204,68,0.3)" if w else "rgba(255,68,68,0.3)"
                  for w in disp["win"]]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["#", "Entry Time", "Exit Time", "Dir", "Entry", "Exit",
                    "Hold", "Exit Reason", "PnL", "Result"],
            fill_color="#0f3460",
            font=dict(color="white", size=12),
            align="center",
            height=32,
        ),
        cells=dict(
            values=[
                disp["trade_num"],
                disp["entry"],
                disp["exit"],
                disp["direction"].str.upper(),
                disp["entry_fmt"],
                disp["exit_fmt"],
                disp["hold_fmt"],
                disp["exit_reason"],
                disp["pnl_fmt"],
                disp["result"],
            ],
            fill_color=[
                ["#16213e"] * len(disp),
                ["#16213e"] * len(disp),
                ["#16213e"] * len(disp),
                ["rgba(0,153,255,0.2)" if d == "long" else "rgba(255,153,0,0.2)"
                 for d in disp["direction"]],
                ["#16213e"] * len(disp),
                ["#16213e"] * len(disp),
                ["#16213e"] * len(disp),
                ["#16213e"] * len(disp),
                pnl_colors,
                pnl_colors,
            ],
            font=dict(color="white", size=11),
            align=["center", "left", "left", "center", "right", "right",
                   "center", "center", "right", "center"],
            height=26,
        ),
    )])
    fig.update_layout(
        title="Full Trade Log — 89 Trades (2024-01 to 2025-06)",
        height=max(500, 30 + len(disp) * 27),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ── main ───────────────────────────────────────────────────────────────────

def build_dashboard(
    trade_log: Path,
    output_html: Path | None = None,
    show: bool = True,
) -> None:
    print("[viz] Loading trade log...")
    trades = _load_trades(trade_log)
    print(f"  {len(trades)} trades loaded")

    print("[viz] Loading price data...")
    price_df = _load_price()

    print("[viz] Building figures...")
    figs = {
        "price":    _fig_price_with_trades(trades, price_df),
        "monthly":  _fig_monthly_pnl(trades),
        "analysis": _fig_trade_analysis(trades),
        "table":    _fig_trade_table(trades),
    }

    if output_html:
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        html_parts = []
        for name, fig in figs.items():
            html_parts.append(f"<h2 style='color:#e0e0e0;font-family:sans-serif;padding:20px 0 0 20px'>{name.upper()}</h2>")
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if name == "price" else False))

        full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Best 4H 43pct Strategy Dashboard</title>
  <style>
    body {{ background: #0d0d1a; margin: 0; padding: 10px; }}
    h1 {{ color: #00cc44; font-family: sans-serif; text-align: center; padding: 20px; }}
    h2 {{ color: #aaa; font-family: sans-serif; }}
  </style>
</head>
<body>
  <h1>XAUUSD Best 4H Strategy — +43.9% | Sharpe 1.87 | Test 2024-2025</h1>
  {''.join(html_parts)}
</body>
</html>"""
        output_html.write_text(full_html, encoding="utf-8")
        print(f"[viz] Saved to {output_html}")

    if show:
        print("[viz] Opening browser...")
        for fig in figs.values():
            fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade log visualizer")
    parser.add_argument("--log",  default="artefacts/best_4h_43pct_trade_log.csv")
    parser.add_argument("--html", default="artefacts/best_4h_43pct_dashboard.html",
                        help="Save combined HTML dashboard (default: artefacts/best_4h_43pct_dashboard.html)")
    parser.add_argument("--no-show", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    build_dashboard(
        trade_log   = PROJECT_ROOT / args.log,
        output_html = PROJECT_ROOT / args.html,
        show        = not args.no_show,
    )
