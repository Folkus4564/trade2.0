"""
components/trade_log.py - Trade Log tab: loads saved *_trades.csv files
from artefacts/backtests/ and renders interactive table + charts.
"""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _load_trade_csvs(backtests_dir: str) -> dict[str, Path]:
    """Return {display_name: path} for all trade CSVs, approved strategies first."""
    result = {}

    # Approved strategies (named, stable) — load first
    approved_dir = Path(backtests_dir).parent / "approved_strategies"
    if approved_dir.exists():
        for strat_dir in sorted(approved_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
            csv = strat_dir / "trades_test.csv"
            if csv.exists():
                result[f"[APPROVED] {strat_dir.name}"] = csv

    # Raw backtest CSVs (may be overwritten each run)
    p = Path(backtests_dir)
    files = sorted(p.glob("*_trades.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files:
        result[f.stem.replace("_trades", "")] = f

    return result


def _equity_from_trades(df: pd.DataFrame, init_cash: float = 100_000) -> pd.DataFrame:
    df = df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("exit_time")
    df["equity"] = df["pnl"].cumsum() + init_cash
    return df.set_index("exit_time")[["equity"]]


def _monthly_pnl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["month"] = df["exit_time"].dt.to_period("M").astype(str)
    return df.groupby("month")["pnl"].sum().reset_index()


def render_trade_log(artefacts_dir: str, live_trades: pd.DataFrame = None) -> None:
    """Render the Trade Log tab."""
    backtests_dir = str(Path(artefacts_dir) / "backtests")
    available = _load_trade_csvs(backtests_dir)

    # ---- Source selector ----
    sources = []
    if live_trades is not None and len(live_trades) > 0:
        sources.append("Current run (live)")
    sources.extend(available.keys())

    if not sources:
        st.info("No trade logs found. Run a backtest (CLI or sidebar) first.")
        return

    selected = st.selectbox("Select trade log", sources)

    if selected == "Current run (live)":
        trades = live_trades.copy()
    else:
        trades = pd.read_csv(available[selected])

    if trades is None or len(trades) == 0:
        st.warning("No trades in this log.")
        return

    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])

    # ---- Summary KPIs ----
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    total_pnl = trades["pnl"].sum()
    win_rate = len(wins) / len(trades)
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

    avg_lots = trades["lots"].mean() if "lots" in trades.columns else None

    cols = st.columns(7)
    cols[0].metric("Trades", len(trades))
    cols[1].metric("Total PnL", f"${total_pnl:,.0f}")
    cols[2].metric("Win Rate", f"{win_rate:.1%}")
    cols[3].metric("Profit Factor", "inf" if pf == float("inf") else f"{pf:.2f}")
    cols[4].metric("Avg Win", f"${avg_win:,.0f}")
    cols[5].metric("Avg Loss", f"${avg_loss:,.0f}")
    cols[6].metric("Avg Lots", f"{avg_lots:.3f}" if avg_lots is not None else "—")

    # ---- Equity curve ----
    eq = _equity_from_trades(trades)
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq.index, y=eq["equity"],
        mode="lines", name="Equity",
        line=dict(color="#00e676", width=2),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
    ))
    fig_eq.update_layout(
        title="Equity Curve", height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="#fafafa", xaxis=dict(gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130"),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ---- Monthly PnL bar chart ----
    monthly = _monthly_pnl(trades)
    colors = ["#00e676" if v >= 0 else "#ff5252" for v in monthly["pnl"]]
    fig_monthly = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["pnl"],
        marker_color=colors, name="Monthly PnL",
    ))
    fig_monthly.update_layout(
        title="Monthly PnL", height=240,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="#fafafa", xaxis=dict(gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", tickprefix="$"),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # ---- Filters ----
    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        directions = ["All"] + sorted(trades["direction"].unique().tolist()) if "direction" in trades.columns else ["All"]
        dir_filter = col1.selectbox("Direction", directions)
        reasons = ["All"] + sorted(trades["exit_reason"].unique().tolist()) if "exit_reason" in trades.columns else ["All"]
        reason_filter = col2.selectbox("Exit Reason", reasons)
        pnl_min, pnl_max = float(trades["pnl"].min()), float(trades["pnl"].max())
        if pnl_min == pnl_max:
            pnl_max = pnl_min + 1.0
        pnl_range = col3.slider("PnL range ($)", pnl_min, pnl_max, (pnl_min, pnl_max))

    filtered = trades.copy()
    if dir_filter != "All" and "direction" in filtered.columns:
        filtered = filtered[filtered["direction"] == dir_filter]
    if reason_filter != "All" and "exit_reason" in filtered.columns:
        filtered = filtered[filtered["exit_reason"] == reason_filter]
    filtered = filtered[(filtered["pnl"] >= pnl_range[0]) & (filtered["pnl"] <= pnl_range[1])]

    # ---- Trade table ----
    st.subheader(f"Trades ({len(filtered)} shown)")

    display = filtered.copy()
    display["entry_time"] = display["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    display["exit_time"] = display["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
    for col in ["entry_price", "exit_price", "sl", "tp"]:
        if col in display.columns:
            display[col] = display[col].round(2)
    if "pnl" in display.columns:
        display["pnl"] = display["pnl"].round(2)
    if "duration_bars" in display.columns:
        display["hold_h"] = (display["duration_bars"] * 5 / 60).round(1)

    show_cols = [c for c in [
        "entry_time", "exit_time", "direction", "lots",
        "entry_price", "sl", "tp", "exit_price",
        "hold_h", "pnl", "exit_reason",
    ] if c in display.columns]

    styled = display[show_cols].style.map(
        lambda v: "color: #00e676" if isinstance(v, (int, float)) and v > 0 else
                  "color: #ff5252" if isinstance(v, (int, float)) and v < 0 else "",
        subset=["pnl"] if "pnl" in show_cols else [],
    )
    st.dataframe(styled, use_container_width=True, height=450)

    # ---- CSV download ----
    csv = display[show_cols].to_csv(index=False)
    st.download_button(
        "Download filtered trades CSV",
        data=csv,
        file_name=f"{selected}_filtered.csv",
        mime="text/csv",
    )
