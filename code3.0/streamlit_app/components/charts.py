"""
components/charts.py - Plotly chart components.
"""
from typing import Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_REGIME_COLORS = {
    "bull":     "rgba(0,200,100,0.08)",
    "bear":     "rgba(220,50,50,0.08)",
    "sideways": "rgba(150,150,150,0.05)",
}

_SOURCE_COLORS = {
    "trend":    "#2196F3",
    "range":    "#FF9800",
    "volatile": "#9E9E9E",
    "cdc":      "#9C27B0",
}

_BOS_MARKERS = {
    "bos_bullish":   ("#00C853", "BOS+"),
    "bos_bearish":   ("#D50000", "BOS-"),
    "choch_bullish": ("#00BFA5", "CHoCH+"),
    "choch_bearish": ("#FF6D00", "CHoCH-"),
}


def candlestick_chart(
    df: pd.DataFrame,
    signals: Optional[pd.DataFrame],
    trades: Optional[pd.DataFrame] = None,
    tail_bars: int = 500,
) -> go.Figure:
    """
    Candlestick chart with trade entry/exit markers, SL/TP lines,
    regime overlay, CDC EMA overlays, LuxAlgo swing points, BOS/CHoCH markers.
    """
    if signals is not None and len(signals) > 0:
        df_plot = signals.tail(tail_bars)
    else:
        df_plot = df.tail(tail_bars)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.02)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"],
        high=df_plot["High"],
        low=df_plot["Low"],
        close=df_plot["Close"],
        name="XAUUSD",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Regime background
    if signals is not None and "regime" in df_plot.columns:
        prev_regime = None
        start_dt = None
        for dt, row in df_plot.iterrows():
            regime = row.get("regime", "sideways")
            if regime != prev_regime:
                if prev_regime is not None and start_dt is not None:
                    fig.add_vrect(
                        x0=start_dt, x1=dt,
                        fillcolor=_REGIME_COLORS.get(prev_regime, "rgba(0,0,0,0)"),
                        layer="below", line_width=0,
                    )
                start_dt = dt
                prev_regime = regime

    # CDC EMA overlays
    if "cdc_fast" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["cdc_fast"],
            name="CDC Fast", line=dict(color="#CE93D8", width=1, dash="dot"),
            hoverinfo="skip",
        ), row=1, col=1)
    if "cdc_slow" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["cdc_slow"],
            name="CDC Slow", line=dict(color="#9C27B0", width=1, dash="dash"),
            hoverinfo="skip",
        ), row=1, col=1)

    # LuxAlgo swing points
    if "swing_high" in df_plot.columns and "swing_high_price" in df_plot.columns:
        sh = df_plot[df_plot["swing_high"].astype(bool)]
        if len(sh) > 0:
            fig.add_trace(go.Scatter(
                x=sh.index, y=sh["swing_high_price"],
                mode="markers",
                marker=dict(symbol="circle-open", size=7, color="#FF6F00", line=dict(width=2)),
                name="Swing High",
                hovertemplate="SH: %{y:.2f}<extra></extra>",
            ), row=1, col=1)

    if "swing_low" in df_plot.columns and "swing_low_price" in df_plot.columns:
        sl_pts = df_plot[df_plot["swing_low"].astype(bool)]
        if len(sl_pts) > 0:
            fig.add_trace(go.Scatter(
                x=sl_pts.index, y=sl_pts["swing_low_price"],
                mode="markers",
                marker=dict(symbol="circle-open", size=7, color="#42A5F5", line=dict(width=2)),
                name="Swing Low",
                hovertemplate="SL: %{y:.2f}<extra></extra>",
            ), row=1, col=1)

    # BOS / CHoCH markers
    for col, (color, label) in _BOS_MARKERS.items():
        if col not in df_plot.columns:
            continue
        bars = df_plot[df_plot[col].astype(bool)]
        if len(bars) == 0:
            continue
        is_bull = label.endswith("+")
        y_vals = bars["High"] * 1.002 if is_bull else bars["Low"] * 0.998
        fig.add_trace(go.Scatter(
            x=bars.index, y=y_vals,
            mode="markers+text",
            marker=dict(symbol="diamond", size=9, color=color, opacity=0.85),
            text=[label] * len(bars),
            textposition="top center" if is_bull else "bottom center",
            textfont=dict(size=8, color=color),
            name=label,
            hovertemplate=f"{label}: %{{x}}<extra></extra>",
        ), row=1, col=1)

    # --- Actual trade entry/exit markers + SL/TP lines ---
    if trades is not None and len(trades) > 0:
        t = trades.copy()
        t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
        t["exit_time"]  = pd.to_datetime(t["exit_time"],  utc=True)

        # Filter to visible window
        t_min = df_plot.index.min()
        t_max = df_plot.index.max()
        vis = t[(t["entry_time"] >= t_min) & (t["entry_time"] <= t_max)]

        for _, tr in vis.iterrows():
            is_long = tr["direction"] == "long"
            win     = tr["pnl"] > 0
            clr_entry = "#00e676" if (is_long and win)  else \
                        "#ff5252" if (is_long and not win) else \
                        "#40c4ff" if (not is_long and win) else "#ffab40"
            clr_exit = "#00e676" if win else "#ff5252"

            # SL line (entry → exit)
            fig.add_shape(
                type="line",
                x0=tr["entry_time"], x1=tr["exit_time"],
                y0=tr["sl"], y1=tr["sl"],
                line=dict(color="rgba(255,82,82,0.55)", width=1, dash="dot"),
                row=1, col=1,
            )
            # TP line (only if set)
            if tr["tp"] > 0 and abs(tr["tp"]) < 1e5:
                fig.add_shape(
                    type="line",
                    x0=tr["entry_time"], x1=tr["exit_time"],
                    y0=tr["tp"], y1=tr["tp"],
                    line=dict(color="rgba(0,230,118,0.45)", width=1, dash="dot"),
                    row=1, col=1,
                )

        # Entry markers
        for direction, symbol, name in [("long", "triangle-up", "Entry Long"), ("short", "triangle-down", "Entry Short")]:
            sub = vis[vis["direction"] == direction]
            if sub.empty:
                continue
            win_mask = sub["pnl"] > 0
            for mask, suffix, opacity in [(win_mask, "Win", 1.0), (~win_mask, "Loss", 0.7)]:
                s = sub[mask]
                if s.empty:
                    continue
                base_color = ("#00e676" if direction == "long" else "#40c4ff") if suffix == "Win" \
                        else ("#ff5252" if direction == "long" else "#ffab40")
                hover = s.apply(lambda r:
                    f"{r.direction.upper()} #{int(r.name)+1 if hasattr(r,'name') else ''}<br>"
                    f"Entry: {r.entry_price:.2f}<br>"
                    f"Exit:  {r.exit_price:.2f}  ({r.exit_reason})<br>"
                    f"PnL:   ${r.pnl:+,.0f}<br>"
                    f"Hold:  {r.duration_bars*5/60:.1f}h", axis=1)
                fig.add_trace(go.Scatter(
                    x=s["entry_time"],
                    y=s["entry_price"],
                    mode="markers",
                    marker=dict(symbol=symbol, size=12, color=base_color,
                                opacity=opacity, line=dict(width=1, color="white")),
                    name=f"{name} {suffix}",
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                ), row=1, col=1)

        # Exit markers (X)
        win_exits  = vis[vis["pnl"] > 0]
        loss_exits = vis[vis["pnl"] <= 0]
        for ex_sub, color, label in [(win_exits, "#00e676", "Exit Win"), (loss_exits, "#ff5252", "Exit Loss")]:
            if ex_sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=ex_sub["exit_time"],
                y=ex_sub["exit_price"],
                mode="markers",
                marker=dict(symbol="x", size=8, color=color,
                            line=dict(width=2, color=color)),
                name=label,
                hovertemplate=f"Exit {label}: %{{y:.2f}}<extra></extra>",
            ), row=1, col=1)

    elif signals is not None:
        # Fallback: raw signal arrows (no trades provided)
        long_entries  = df_plot[df_plot["signal_long"]  == 1] if "signal_long"  in df_plot.columns else pd.DataFrame()
        short_entries = df_plot[df_plot["signal_short"] == 1] if "signal_short" in df_plot.columns else pd.DataFrame()
        for source, color in _SOURCE_COLORS.items():
            if "signal_source" in df_plot.columns:
                le = long_entries[long_entries["signal_source"] == source] if len(long_entries) > 0 else pd.DataFrame()
                se = short_entries[short_entries["signal_source"] == source] if len(short_entries) > 0 else pd.DataFrame()
            else:
                le = long_entries if source == "trend" else pd.DataFrame()
                se = short_entries if source == "trend" else pd.DataFrame()
            if len(le) > 0:
                fig.add_trace(go.Scatter(
                    x=le.index, y=le["Low"] * 0.999, mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color=color),
                    name=f"Long ({source})",
                ), row=1, col=1)
            if len(se) > 0:
                fig.add_trace(go.Scatter(
                    x=se.index, y=se["High"] * 1.001, mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color=color),
                    name=f"Short ({source})",
                ), row=1, col=1)

    # Regime probability stacked area (row 2)
    prob_cols = {
        "bull_prob":     ("rgba(0,200,100,0.6)",  "rgba(0,200,100,0.25)",  "Bull Prob"),
        "bear_prob":     ("rgba(220,50,50,0.6)",   "rgba(220,50,50,0.25)",  "Bear Prob"),
        "sideways_prob": ("rgba(150,150,150,0.6)", "rgba(150,150,150,0.2)", "Sideways Prob"),
    }
    if signals is not None and any(c in df_plot.columns for c in prob_cols):
        for col, (line_color, fill_color, label) in prob_cols.items():
            if col not in df_plot.columns:
                continue
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot[col],
                stackgroup="prob", name=label,
                line=dict(color=line_color, width=0),
                fillcolor=fill_color,
                hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
            ), row=2, col=1)

    fig.update_layout(
        title="XAUUSD Price + Trades",
        xaxis_rangeslider_visible=False,
        height=700,
        template="plotly_dark",
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="HMM Prob", row=2, col=1)
    return fig


def equity_curve_chart(results: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    """Equity curve vs buy-and-hold with drawdown area."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.05)

    if results is None or "equity" not in results.columns:
        fig.add_annotation(text="No results available", x=0.5, y=0.5, showarrow=False)
        return fig

    equity = results["equity"]
    peak   = equity.cummax()
    dd     = (equity - peak) / peak

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity,
        name="Strategy", line=dict(color="#2196F3", width=2),
    ), row=1, col=1)

    if "Close" in results.columns:
        bh = results["Close"] / results["Close"].iloc[0] * equity.iloc[0]
        fig.add_trace(go.Scatter(
            x=bh.index, y=bh,
            name="Buy & Hold", line=dict(color="#9E9E9E", width=1, dash="dash"),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd * 100,
        fill="tozeroy", name="Drawdown %",
        line=dict(color="rgba(220,50,50,0.8)", width=1),
        fillcolor="rgba(220,50,50,0.3)",
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(title=title, height=550, template="plotly_dark")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Equity ($)",    row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig


def monthly_pnl_chart(trades: pd.DataFrame) -> go.Figure:
    """Monthly PnL bar chart with win/total labels."""
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t["win"] = t["pnl"] > 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t["month"] = t["exit_time"].dt.to_period("M").dt.to_timestamp()

    monthly = t.groupby("month").agg(
        pnl=("pnl", "sum"),
        trades=("pnl", "count"),
        wins=("win", "sum"),
    ).reset_index()
    monthly["label"]  = monthly["month"].dt.strftime("%b %Y")
    monthly["colors"] = monthly["pnl"].apply(lambda x: "#00e676" if x >= 0 else "#ff5252")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["label"],
        y=monthly["pnl"],
        marker_color=monthly["colors"],
        text=monthly.apply(lambda r: f"${r.pnl:+,.0f}<br>{int(r.wins)}/{int(r.trades)}", axis=1),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>PnL: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="white", width=1, dash="dot"))
    fig.update_layout(
        title="Monthly PnL  (wins / total on bars)",
        height=380, template="plotly_dark",
        yaxis_title="PnL ($)", xaxis_title="Month",
        margin=dict(t=50, b=60),
    )
    return fig


def trade_analysis_chart(trades: pd.DataFrame) -> go.Figure:
    """3-panel: PnL waterfall | hold duration histogram | exit reason bars."""
    t = trades.copy()
    t["win"] = t["pnl"] > 0
    t["hold_hours"] = t["duration_bars"] * 5 / 60
    t["trade_num"]  = range(1, len(t) + 1)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("PnL per Trade", "Hold Duration (hours)", "Exit Reasons"),
        horizontal_spacing=0.10,
    )

    # 1. PnL waterfall
    colors = ["#00e676" if p > 0 else "#ff5252" for p in t["pnl"]]
    fig.add_trace(go.Bar(
        x=t["trade_num"], y=t["pnl"],
        marker_color=colors, name="Trade PnL",
        hovertemplate="Trade #%{x}<br>PnL: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # 2. Duration histogram
    fig.add_trace(go.Histogram(
        x=t["hold_hours"], nbinsx=15,
        marker_color="#40c4ff", name="Hold Hours",
        hovertemplate="Hold: %{x:.0f}h<br>Count: %{y}<extra></extra>",
    ), row=1, col=2)

    # 3. Exit reason bar
    reason_counts = t["exit_reason"].value_counts().reset_index()
    reason_counts.columns = ["reason", "count"]
    exit_colors = {"tp": "#00e676", "sl": "#ff5252", "signal": "#40c4ff",
                   "end_of_data": "#ffab40"}
    fig.add_trace(go.Bar(
        x=reason_counts["reason"],
        y=reason_counts["count"],
        marker_color=[exit_colors.get(r, "#9e9e9e") for r in reason_counts["reason"]],
        text=reason_counts["count"],
        textposition="outside",
        name="Exit Reason",
        hovertemplate="%{x}: %{y}<extra></extra>",
    ), row=1, col=3)

    fig.update_layout(
        height=380, template="plotly_dark",
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    fig.update_xaxes(title_text="Trade #",    row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)",    row=1, col=1)
    fig.update_xaxes(title_text="Hours",      row=1, col=2)
    fig.update_yaxes(title_text="# Trades",   row=1, col=2)
    fig.update_xaxes(title_text="Reason",     row=1, col=3)
    fig.update_yaxes(title_text="# Trades",   row=1, col=3)
    return fig
