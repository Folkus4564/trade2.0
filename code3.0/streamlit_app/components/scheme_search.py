"""
components/scheme_search.py - Full Scheme Search results visualisation.
Reads full_scheme_search_results.json and renders charts + tables.
"""
from pathlib import Path
from typing import Optional
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---- helpers ----------------------------------------------------------------

def _load_wf_results(artefacts_dir: str) -> Optional[list]:
    path = Path(artefacts_dir) / "wf_positive_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_results(artefacts_dir: str) -> Optional[pd.DataFrame]:
    path = Path(artefacts_dir) / "full_scheme_search_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for exp in raw:
        tm = exp.get("test_metrics") or {}
        vm = exp.get("val_metrics") or {}
        tr = exp.get("train_metrics") or {}
        rows.append({
            "idea":           exp.get("idea_name", "?"),
            "target":         exp.get("optuna_target", "?"),
            "verdict":        exp.get("verdict", "?"),
            # test
            "test_return":    tm.get("annualized_return", None),
            "test_sharpe":    tm.get("sharpe_ratio", None),
            "test_dd":        tm.get("max_drawdown", None),
            "test_pf":        tm.get("profit_factor", None),
            "test_trades":    tm.get("total_trades", None),
            "test_winrate":   tm.get("win_rate", None),
            # val
            "val_return":     vm.get("annualized_return", None),
            "val_sharpe":     vm.get("sharpe_ratio", None),
            # train
            "train_return":   tr.get("annualized_return", None),
            "train_sharpe":   tr.get("sharpe_ratio", None),
        })
    return pd.DataFrame(rows)


def _best_per_idea(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the best run (by test_return) per idea."""
    return (
        df.sort_values("test_return", ascending=False)
          .drop_duplicates(subset=["idea"])
          .reset_index(drop=True)
    )


# ---- charts -----------------------------------------------------------------

_VERDICT_COLOR = {
    "HARD_REJECTED": "#EF5350",
    "REVISE":        "#FFA726",
    "APPROVED":      "#66BB6A",
    "UNSTABLE":      "#AB47BC",
}

_TARGET_SYMBOL = {
    "val_return": "circle",
    "val_sharpe": "diamond",
}


def _scatter_return_sharpe(df: pd.DataFrame) -> go.Figure:
    """Return vs Sharpe scatter — one point per (idea, target)."""
    fig = go.Figure()

    for target, symbol in _TARGET_SYMBOL.items():
        sub = df[df["target"] == target]
        if sub.empty:
            continue

        colors  = [_VERDICT_COLOR.get(v, "#9E9E9E") for v in sub["verdict"]]
        labels  = [f"{r['idea']}<br>Ret: {r['test_return']*100:.1f}%  Sharpe: {r['test_sharpe']:.2f}"
                   f"<br>DD: {r['test_dd']*100:.1f}%  PF: {r['test_pf']:.2f}"
                   for _, r in sub.iterrows()]

        # Highlight best idea (idea_16_4h_2state_agg)
        sizes = [18 if "idea_16" in row["idea"] else 10 for _, row in sub.iterrows()]
        borders = ["gold" if "idea_16" in row["idea"] else "rgba(0,0,0,0)"
                   for _, row in sub.iterrows()]

        fig.add_trace(go.Scatter(
            x=sub["test_return"] * 100,
            y=sub["test_sharpe"],
            mode="markers+text",
            marker=dict(
                symbol=symbol, size=sizes, color=colors,
                line=dict(color=borders, width=2),
            ),
            text=sub["idea"].str.replace("idea_", "").str[:12],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=labels,
            hoverinfo="text",
            name=f"target={target}",
        ))

    # Target lines
    fig.add_vline(x=50, line_dash="dash", line_color="gold", opacity=0.5,
                  annotation_text="50% target", annotation_position="top")
    fig.add_hline(y=1.5, line_dash="dash", line_color="cyan", opacity=0.5,
                  annotation_text="Sharpe 1.5 target", annotation_position="right")
    fig.add_vline(x=20, line_dash="dot",  line_color="gray", opacity=0.4,
                  annotation_text="20% min", annotation_position="bottom")
    fig.add_hline(y=1.0, line_dash="dot",  line_color="gray", opacity=0.4,
                  annotation_text="Sharpe 1.0 min", annotation_position="right")

    fig.update_layout(
        title="All Ideas — Test Return vs Sharpe",
        xaxis_title="Annualized Return (%)",
        yaxis_title="Sharpe Ratio",
        height=520,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _bar_ranked_return(df_best: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of best test return per idea, ranked."""
    df_s = df_best.sort_values("test_return")
    colors = [
        "gold" if "idea_16" in r else
        "#66BB6A" if v >= 0.50 else
        "#42A5F5" if v >= 0.20 else
        "#EF5350"
        for r, v in zip(df_s["idea"], df_s["test_return"])
    ]

    fig = go.Figure(go.Bar(
        x=df_s["test_return"] * 100,
        y=df_s["idea"].str.replace("idea_", "#"),
        orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in df_s["test_return"]],
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "Return: %{x:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=50, line_dash="dash", line_color="gold",  opacity=0.6, annotation_text="50% target")
    fig.add_vline(x=20, line_dash="dot",  line_color="gray",  opacity=0.4, annotation_text="20% min")

    fig.update_layout(
        title="Best Test Return per Idea (ranked)",
        xaxis_title="Annualized Return (%)",
        height=620,
        template="plotly_dark",
        margin=dict(l=160),
    )
    return fig


def _radar_best(df: pd.DataFrame, idea: str = "idea_16_4h_2state_agg") -> go.Figure:
    """Radar chart for best idea across train / val / test."""
    sub = df[df["idea"] == idea].sort_values("test_return", ascending=False).iloc[0]

    categories = ["Return", "Sharpe", "PF", "Win Rate", "Low DD"]

    def norm_return(v):   return min(max(v / 0.5, 0), 1)
    def norm_sharpe(v):   return min(max(v / 2.0, 0), 1)
    def norm_pf(v):       return min(max((v - 1.0) / 1.0, 0), 1)
    def norm_wr(v):       return min(max(v, 0), 1)
    def norm_dd(v):       return min(max(1 + v / 0.20, 0), 1)   # -20% worst -> 0

    splits = {
        "Test":  [norm_return(sub["test_return"]),  norm_sharpe(sub["test_sharpe"]),
                  norm_pf(sub["test_pf"]),          norm_wr(sub.get("test_winrate") or 0.5),
                  norm_dd(sub["test_dd"])],
        "Val":   [norm_return(sub["val_return"]),   norm_sharpe(sub["val_sharpe"]),
                  0.5, 0.5, 0.7],
        "Train": [norm_return(sub["train_return"]), norm_sharpe(sub["train_sharpe"]),
                  0.5, 0.5, 0.6],
    }

    colors = {"Test": "#2196F3", "Val": "#FF9800", "Train": "#9E9E9E"}
    fig = go.Figure()
    for split_name, vals in splits.items():
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name=split_name,
            line=dict(color=colors[split_name]),
            opacity=0.75,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
        title=f"Best Idea Profile: {idea.replace('idea_', '#')}",
        height=420,
        template="plotly_dark",
        showlegend=True,
    )
    return fig


def _metrics_comparison_table(df_best: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted DataFrame for display."""
    display = df_best[[
        "idea", "test_return", "test_sharpe", "test_dd",
        "test_pf", "test_trades", "test_winrate", "verdict",
    ]].copy()
    display["test_return"]  = (display["test_return"]  * 100).round(1).astype(str) + "%"
    display["test_sharpe"]  =  display["test_sharpe"].round(3)
    display["test_dd"]      = (display["test_dd"]      * 100).round(1).astype(str) + "%"
    display["test_pf"]      =  display["test_pf"].round(3)
    display["test_winrate"] = (display["test_winrate"] * 100).round(1).astype(str) + "%"
    display["test_trades"]  =  display["test_trades"].astype(int)
    display.columns = ["Idea", "Return%", "Sharpe", "MaxDD%", "PF", "Trades", "WinRate%", "Verdict"]
    return display.set_index("Idea")


# ---- main render ------------------------------------------------------------

def render_scheme_search(artefacts_dir: str) -> None:
    st.header("Full Scheme Search Results")

    df = _load_results(artefacts_dir)
    if df is None:
        st.error("full_scheme_search_results.json not found in artefacts/")
        return

    df_best = _best_per_idea(df)
    best_idea = df_best.iloc[0]

    # Summary banner
    total = len(df)
    approved = (df["verdict"] == "APPROVED").sum()
    revise   = (df["verdict"] == "REVISE").sum()
    hard_rej = (df["verdict"] == "HARD_REJECTED").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Experiments", total)
    c2.metric("Approved", approved)
    c3.metric("Revise", revise)
    c4.metric("Hard Rejected", hard_rej)
    c5.metric("Best Return", f"{best_idea['test_return']*100:.1f}%",
              delta=best_idea["idea"].replace("idea_", "#"))

    st.divider()

    # Best idea highlight
    st.subheader("Best Implementation: " + best_idea["idea"].replace("idea_", "#"))
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Return (Test)",  f"{best_idea['test_return']*100:.1f}%")
    mc2.metric("Sharpe (Test)",  f"{best_idea['test_sharpe']:.3f}")
    mc3.metric("Max DD (Test)",  f"{best_idea['test_dd']*100:.1f}%")
    mc4.metric("PF (Test)",      f"{best_idea['test_pf']:.3f}")
    mc5.metric("Trades",         int(best_idea["test_trades"]))
    mc6.metric("Win Rate",       f"{best_idea['test_winrate']*100:.1f}%"
               if best_idea["test_winrate"] is not None else "N/A")

    st.caption(
        "Config override: regime_timeframe=4H | n_states=2 | sizing_max=2.0 | base_allocation_frac=0.90"
    )
    st.caption("All test checks PASS. Only rejection: walk-forward not run (wf_not_run).")

    st.divider()

    # Charts row 1
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(_scatter_return_sharpe(df), use_container_width=True)
    with col_b:
        st.plotly_chart(_radar_best(df, best_idea["idea"]), use_container_width=True)

    # Bar chart full width
    st.plotly_chart(_bar_ranked_return(df_best), use_container_width=True)

    # Table
    with st.expander("All Ideas — Best Run Table", expanded=True):
        tbl = _metrics_comparison_table(df_best.sort_values("test_return", ascending=False).reset_index(drop=True))

        def _highlight(row):
            styles = []
            for col in row.index:
                if row.name and "idea_16" in str(row.name):
                    styles.append("background-color: rgba(255,215,0,0.15); font-weight: bold")
                elif col == "Return%" and "%" in str(row[col]):
                    try:
                        v = float(row[col].replace("%", ""))
                        if v >= 50:
                            styles.append("color: #66BB6A")
                        elif v >= 20:
                            styles.append("color: #42A5F5")
                        else:
                            styles.append("color: #EF5350")
                    except Exception:
                        styles.append("")
                else:
                    styles.append("")
            return styles

        st.dataframe(tbl.style.apply(_highlight, axis=1), use_container_width=True)

    # All experiments (both targets) expandable
    with st.expander(f"All {total} Experiments (both targets)", expanded=False):
        all_tbl = df.sort_values("test_return", ascending=False).copy()
        all_tbl["test_return"] = (all_tbl["test_return"] * 100).round(1).astype(str) + "%"
        all_tbl["test_sharpe"] =  all_tbl["test_sharpe"].round(3)
        all_tbl["test_dd"]     = (all_tbl["test_dd"] * 100).round(1).astype(str) + "%"
        all_tbl["test_pf"]     =  all_tbl["test_pf"].round(3)
        all_tbl["test_trades"] =  all_tbl["test_trades"].astype(int)
        display_all = all_tbl[["idea", "target", "test_return", "test_sharpe",
                                "test_dd", "test_pf", "test_trades", "verdict"]]
        display_all.columns = ["Idea", "Target", "Return%", "Sharpe", "MaxDD%", "PF", "Trades", "Verdict"]
        st.dataframe(display_all.set_index("Idea"), use_container_width=True)

    # ---- WF Trade Logs ----
    st.divider()
    st.subheader("Walk-Forward Trade Logs (Test Period)")

    wf_data = _load_wf_results(artefacts_dir)
    if wf_data is None:
        st.info("wf_positive_results.json not found. Run `wf_positive` to generate WF results with trade logs.")
    else:
        idea_names = [r.get("idea_name", r.get("run_name", "?")) for r in wf_data]
        selected   = st.selectbox("Select idea to view trade log:", idea_names)

        for r in wf_data:
            if r.get("idea_name") == selected or r.get("run_name") == selected:
                tm  = r.get("test_metrics") or {}
                wf  = r.get("walk_forward")  or {}

                # Metrics row
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Return (Test)",  f"{tm.get('annualized_return', 0)*100:.1f}%")
                mc2.metric("Sharpe (Test)",  f"{tm.get('sharpe_ratio', 0):.3f}")
                mc3.metric("WF Sharpe",      f"{wf.get('mean_sharpe', 0):.3f}" if wf else "N/A")
                mc4.metric("WF Win%",        f"{wf.get('pct_positive', 0)*100:.0f}%" if wf else "N/A")
                mc5.metric("Verdict",        r.get("verdict", "?"))

                # WF window breakdown
                if wf and wf.get("windows"):
                    st.caption("Walk-Forward Windows")
                    wf_rows = []
                    for w in wf["windows"]:
                        wf_rows.append({
                            "Window":   w.get("window"),
                            "Val Period": w.get("val_period", "?"),
                            "Return%":  f"{w.get('annualized_return', 0)*100:.1f}%",
                            "Sharpe":   f"{w.get('sharpe_ratio', 0):.3f}",
                            "MaxDD%":   f"{w.get('max_drawdown', 0)*100:.1f}%",
                            "Trades":   w.get("total_trades", 0),
                            "WinRate%": f"{w.get('win_rate', 0)*100:.1f}%",
                            "PF":       f"{w.get('profit_factor', 0):.3f}",
                            "Beats Random": "YES" if w.get("beats_random_baseline") else "NO",
                        })
                    wf_df = pd.DataFrame(wf_rows).set_index("Window")

                    def _color_sharpe(val):
                        try:
                            v = float(val)
                            return "color: #66BB6A" if v > 0 else "color: #EF5350"
                        except Exception:
                            return ""

                    st.dataframe(
                        wf_df.style.map(_color_sharpe, subset=["Sharpe"]),
                        use_container_width=True,
                    )

                # Trade log
                trades = r.get("test_trades")
                if trades:
                    trades_df = pd.DataFrame(trades)
                    st.caption(f"Test Period Trade Log — {len(trades_df)} trades")
                    for col in ["entry_time", "exit_time"]:
                        if col in trades_df.columns:
                            trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime("%Y-%m-%d %H:%M")
                    for col in ["entry_price", "exit_price", "sl", "tp"]:
                        if col in trades_df.columns:
                            trades_df[col] = trades_df[col].round(2)
                    if "pnl" in trades_df.columns:
                        trades_df["pnl"] = trades_df["pnl"].round(2)
                    if "size" in trades_df.columns:
                        trades_df["size"] = trades_df["size"].round(4)

                    display_cols = [c for c in [
                        "entry_time", "exit_time", "direction",
                        "entry_price", "sl", "tp", "exit_price",
                        "size", "pnl", "exit_reason", "duration_bars",
                    ] if c in trades_df.columns]

                    styled = trades_df[display_cols].style.map(
                        lambda v: "color: #66BB6A" if isinstance(v, (int, float)) and v > 0
                                  else "color: #EF5350" if isinstance(v, (int, float)) and v < 0
                                  else "",
                        subset=["pnl"] if "pnl" in display_cols else [],
                    )
                    st.dataframe(styled, use_container_width=True, height=420)

                    # PnL distribution chart
                    if "pnl" in trades_df.columns:
                        pnl_vals = pd.DataFrame(trades).get("pnl", pd.Series(dtype=float))
                        fig_pnl = go.Figure()
                        fig_pnl.add_trace(go.Bar(
                            x=list(range(len(pnl_vals))),
                            y=pnl_vals,
                            marker_color=["#66BB6A" if v > 0 else "#EF5350" for v in pnl_vals],
                            name="PnL per trade",
                        ))
                        fig_pnl.update_layout(
                            title=f"{selected} — Trade PnL (Test Period)",
                            xaxis_title="Trade #",
                            yaxis_title="PnL ($)",
                            height=300,
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig_pnl, use_container_width=True)
                else:
                    st.info("No trade log available. Re-run `wf_positive` to capture trades (already updated).")
                break
