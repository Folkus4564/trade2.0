"""
components/charts.py - Plotly chart components.
"""
from typing import Dict, Any, Optional
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


def candlestick_chart(
    df: pd.DataFrame,
    signals: Optional[pd.DataFrame],
    tail_bars: int = 500,
) -> go.Figure:
    """
    Candlestick chart with trade entry/exit markers and regime overlay.

    Args:
        df:        OHLCV DataFrame (5M or 1H).
        signals:   Signal DataFrame with signal_long, signal_short, signal_source.
        tail_bars: Number of recent bars to display.
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

    # Trade markers
    if signals is not None:
        long_entries  = df_plot[df_plot["signal_long"]  == 1] if "signal_long"  in df_plot.columns else pd.DataFrame()
        short_entries = df_plot[df_plot["signal_short"] == 1] if "signal_short" in df_plot.columns else pd.DataFrame()

        for source, color in _SOURCE_COLORS.items():
            le = long_entries[long_entries.get("signal_source", pd.Series()) == source] if len(long_entries) > 0 else pd.DataFrame()
            se = short_entries[short_entries.get("signal_source", pd.Series()) == source] if len(short_entries) > 0 else pd.DataFrame()

            if len(le) > 0:
                fig.add_trace(go.Scatter(
                    x=le.index, y=le["Low"] * 0.999,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color=color),
                    name=f"Long ({source})",
                ), row=1, col=1)

            if len(se) > 0:
                fig.add_trace(go.Scatter(
                    x=se.index, y=se["High"] * 1.001,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color=color),
                    name=f"Short ({source})",
                ), row=1, col=1)

    # Regime probability stacked area (row 2)
    if signals is not None and all(c in df_plot.columns for c in ["bull_prob", "bear_prob", "sideways_prob"]):
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["bull_prob"],
            fill="tozeroy", name="Bull Prob",
            line=dict(color="rgba(0,200,100,0.8)", width=1),
            fillcolor="rgba(0,200,100,0.3)",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["bear_prob"],
            fill="tozeroy", name="Bear Prob",
            line=dict(color="rgba(220,50,50,0.8)", width=1),
            fillcolor="rgba(220,50,50,0.3)",
        ), row=2, col=1)

    fig.update_layout(
        title="XAUUSD Price + Signals",
        xaxis_rangeslider_visible=False,
        height=700,
        template="plotly_dark",
    )
    return fig


def equity_curve_chart(results: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    """
    Equity curve vs buy-and-hold with drawdown area.

    Args:
        results: Backtest results DataFrame with 'equity' column.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.05)

    if results is None or "equity" not in results.columns:
        fig.add_annotation(text="No results available", x=0.5, y=0.5, showarrow=False)
        return fig

    equity = results["equity"]
    peak   = equity.cummax()
    dd     = (equity - peak) / peak

    # Equity line
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity,
        name="Strategy", line=dict(color="#2196F3", width=2),
    ), row=1, col=1)

    # Buy-and-hold if close available
    if "Close" in results.columns:
        bh = results["Close"] / results["Close"].iloc[0] * equity.iloc[0]
        fig.add_trace(go.Scatter(
            x=bh.index, y=bh,
            name="Buy & Hold", line=dict(color="#9E9E9E", width=1, dash="dash"),
        ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd * 100,
        fill="tozeroy", name="Drawdown %",
        line=dict(color="rgba(220,50,50,0.8)", width=1),
        fillcolor="rgba(220,50,50,0.3)",
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=title,
        height=550,
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig
