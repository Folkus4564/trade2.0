"""
components/approved_strategies.py - Approved Strategies section.
Loads all strategy exports from artefacts/approved_strategies/ and
renders a comparison table + detail view per strategy.
"""
from pathlib import Path
import json
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _load_strategies(approved_dir: str) -> list[dict]:
    """Load all metrics.json files from approved_strategies subdirs, newest first."""
    base = Path(approved_dir)
    strategies = []
    for subdir in sorted(base.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        metrics_path = subdir / "metrics.json"
        summary_path = subdir / "training_summary.md"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        data["_dir"] = str(subdir)
        data["_name"] = subdir.name
        data["_summary"] = summary_path.read_text() if summary_path.exists() else None
        trades_path = subdir / "trades_test.csv"
        data["_trades_path"] = str(trades_path) if trades_path.exists() else None
        strategies.append(data)
    return strategies


def _fmt(val, pct=False, dp=2):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    if pct:
        return f"{val:.1%}"
    return f"{val:.{dp}f}"


def _verdict_badge(verdict: str) -> str:
    if verdict == "APPROVED":
        return ":white_check_mark: APPROVED"
    return ":x: HARD_REJECTED"


def _split_card(label: str, m: dict) -> None:
    if not m:
        st.caption(f"{label}: no data")
        return
    st.caption(label)
    c1, c2 = st.columns(2)
    c1.metric("Return", _fmt(m.get("annualized_return"), pct=True))
    c2.metric("Sharpe", _fmt(m.get("sharpe_ratio")))
    c3, c4 = st.columns(2)
    c3.metric("Max DD", _fmt(m.get("max_drawdown"), pct=True))
    c4.metric("Trades", m.get("total_trades", "—"))
    c5, c6 = st.columns(2)
    c5.metric("Win Rate", _fmt(m.get("win_rate"), pct=True))
    c6.metric("Profit Factor", _fmt(m.get("profit_factor")))


def render_approved_strategies(artefacts_dir: str) -> None:
    approved_dir = str(Path(artefacts_dir) / "approved_strategies")
    strategies = _load_strategies(approved_dir)

    if not strategies:
        st.info("No approved strategies found. Run `trade2 --export-approved` to save one.")
        return

    # ---- Comparison table ----
    st.subheader(f"Saved Strategies ({len(strategies)})")

    import pandas as pd
    rows = []
    for s in strategies:
        test = s.get("test") or {}
        wf = s.get("walk_forward") or {}
        rows.append({
            "Name": s.get("_name", s.get("strategy_name", "?")),
            "Date": s.get("date", "?"),
            "Verdict": s.get("verdict", "?"),
            "Test Return": _fmt(test.get("annualized_return"), pct=True),
            "Test Sharpe": _fmt(test.get("sharpe_ratio")),
            "Test DD": _fmt(test.get("max_drawdown"), pct=True),
            "Win Rate": _fmt(test.get("win_rate"), pct=True),
            "PF": _fmt(test.get("profit_factor")),
            "Trades": test.get("total_trades", "—"),
            "WF Mean Sharpe": _fmt(wf.get("mean_sharpe")) if wf else "—",
            "WF Win%": _fmt(wf.get("positive_windows_pct"), pct=True) if wf and wf.get("positive_windows_pct") else "—",
        })
    df = pd.DataFrame(rows)

    def color_verdict(val):
        if val == "APPROVED":
            return "color: #00e676; font-weight: bold"
        if "REJECTED" in str(val):
            return "color: #ff5252"
        return ""

    styled = df.style.map(color_verdict, subset=["Verdict"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Detail view ----
    names = [s.get("_name", s.get("strategy_name", f"strategy_{i}")) for i, s in enumerate(strategies)]
    selected_name = st.selectbox("Inspect strategy", names)
    s = next(x for x in strategies if x.get("_name", x.get("strategy_name")) == selected_name)

    verdict = s.get("verdict", "?")
    st.markdown(f"### {selected_name}  {_verdict_badge(verdict)}")
    st.caption(f"Saved: {s.get('date', '?')}")

    # Hard rejection reasons
    hard = s.get("hard_checks") or {}
    if hard.get("hard_rejected") and hard.get("rejections"):
        with st.container():
            st.error("**Hard Rejection Reasons:**")
            for key, reason in hard["rejections"].items():
                st.markdown(f"- `{key}`: {reason}")

    # Train / Val / Test cards
    st.markdown("#### Performance by Split")
    col_tr, col_va, col_te = st.columns(3)
    with col_tr:
        _split_card("TRAIN", s.get("train"))
    with col_va:
        _split_card("VAL", s.get("val"))
    with col_te:
        _split_card("TEST", s.get("test"))
        # 2x cost sensitivity
        test_m = s.get("test") or {}
        cost2x = test_m.get("cost_sensitivity_2x")
        if cost2x:
            st.caption("2x Cost Sensitivity")
            st.metric("Sharpe @ 2x", _fmt(cost2x.get("sharpe_ratio")))
            st.metric("Return @ 2x", _fmt(cost2x.get("annualized_return"), pct=True))

    # Walk-forward
    wf = s.get("walk_forward")
    if wf:
        st.markdown("#### Walk-Forward Results")
        w1, w2, w3 = st.columns(3)
        w1.metric("Mean Sharpe", _fmt(wf.get("mean_sharpe")))
        w2.metric("Positive Windows", _fmt(wf.get("positive_windows_pct"), pct=True) if wf.get("positive_windows_pct") else f"{wf.get('positive_windows','?')}/{wf.get('n_windows','?')}")
        w3.metric("Windows", wf.get("n_windows", wf.get("windows", "?")))

    # Parameters
    params = s.get("params")
    if params:
        with st.expander("Parameters", expanded=False):
            pcols = st.columns(3)
            for i, (k, v) in enumerate(params.items()):
                pcols[i % 3].metric(k, v if isinstance(v, bool) else (f"{v:.4f}" if isinstance(v, float) else str(v)))

    # Training summary markdown
    summary = s.get("_summary")
    if summary:
        with st.expander("Training Summary", expanded=False):
            st.markdown(summary)

    # Trade log (test split only)
    trades_path = s.get("_trades_path")
    if trades_path:
        st.markdown("#### Trade Log (Test Split)")
        trades = pd.read_csv(trades_path)
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
        trades["exit_time"] = pd.to_datetime(trades["exit_time"])

        # KPIs
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        total_pnl = trades["pnl"].sum()
        win_rate = len(wins) / len(trades)
        pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
        kc = st.columns(5)
        kc[0].metric("Total Trades", len(trades))
        kc[1].metric("Total PnL", f"${total_pnl:,.0f}")
        kc[2].metric("Win Rate", f"{win_rate:.1%}")
        kc[3].metric("Profit Factor", "inf" if pf == float("inf") else f"{pf:.2f}")
        kc[4].metric("Avg PnL / Trade", f"${trades['pnl'].mean():,.0f}")

        # Equity curve
        eq = trades.sort_values("exit_time").copy()
        eq["equity"] = eq["pnl"].cumsum() + 100_000
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq["exit_time"], y=eq["equity"],
            mode="lines", name="Equity",
            line=dict(color="#00e676", width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
        ))
        fig_eq.update_layout(
            title="Equity Curve (Test)", height=260,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#fafafa",
            xaxis=dict(gridcolor="#1e2130"),
            yaxis=dict(gridcolor="#1e2130", tickprefix="$"),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Monthly PnL
        monthly = trades.copy()
        monthly["month"] = monthly["exit_time"].dt.to_period("M").astype(str)
        monthly = monthly.groupby("month")["pnl"].sum().reset_index()
        colors = ["#00e676" if v >= 0 else "#ff5252" for v in monthly["pnl"]]
        fig_monthly = go.Figure(go.Bar(
            x=monthly["month"], y=monthly["pnl"],
            marker_color=colors, name="Monthly PnL",
        ))
        fig_monthly.update_layout(
            title="Monthly PnL", height=220,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#fafafa",
            xaxis=dict(gridcolor="#1e2130"),
            yaxis=dict(gridcolor="#1e2130", tickprefix="$"),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Trade table
        with st.expander(f"All {len(trades)} Trades", expanded=True):
            display = trades.copy()
            display["entry_time"] = display["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
            display["exit_time"] = display["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
            for col in ["entry_price", "exit_price", "sl", "tp"]:
                if col in display.columns:
                    display[col] = display[col].round(2)
            display["pnl"] = display["pnl"].round(2)
            if "duration_bars" in display.columns:
                display["hold_h"] = (display["duration_bars"] * 5 / 60).round(1)
            show_cols = [c for c in [
                "entry_time", "exit_time", "direction",
                "entry_price", "sl", "tp", "exit_price",
                "hold_h", "pnl", "exit_reason",
            ] if c in display.columns]
            styled = display[show_cols].style.map(
                lambda v: "color: #00e676" if isinstance(v, (int, float)) and v > 0 else
                          "color: #ff5252" if isinstance(v, (int, float)) and v < 0 else "",
                subset=["pnl"] if "pnl" in show_cols else [],
            )
            st.dataframe(styled, use_container_width=True, height=420)

        csv = display[show_cols].to_csv(index=False)
        st.download_button(
            "Download trade log CSV",
            data=csv,
            file_name=f"{selected_name}_trades_test.csv",
            mime="text/csv",
        )
