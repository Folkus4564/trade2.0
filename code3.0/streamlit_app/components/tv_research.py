"""
components/tv_research.py - TV Indicator Research results visualisation.
Reads tv_research_log.json from artefacts/tv_research/ and renders charts + tables.
"""
from pathlib import Path
from typing import Optional
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_log(artefacts_dir: str) -> Optional[pd.DataFrame]:
    path = Path(artefacts_dir) / "tv_research" / "tv_research_log.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for entry in raw:
        tm = entry.get("test_metrics") or {}
        bm = entry.get("baseline_metrics") or {}
        dv = entry.get("delta_vs_baseline") or {}
        rows.append({
            "id":           entry.get("id", 0),
            "name":         entry.get("name", "?"),
            "category":     entry.get("category", "?"),
            "source":       entry.get("source", "?"),
            "status":       entry.get("status", "?"),
            "verdict":      entry.get("verdict", "?"),
            "timestamp":    entry.get("timestamp", ""),
            "duration_s":   entry.get("duration_seconds", 0),
            "is_best":      entry.get("is_best", False),
            # test
            "test_return":  tm.get("annualized_return"),
            "test_sharpe":  tm.get("sharpe_ratio"),
            "test_dd":      tm.get("max_drawdown"),
            "test_pf":      tm.get("profit_factor"),
            "test_trades":  tm.get("total_trades"),
            "test_wr":      tm.get("win_rate"),
            # deltas
            "delta_return": dv.get("return_delta"),
            "delta_sharpe": dv.get("sharpe_delta"),
            # baseline
            "base_return":  bm.get("annualized_return"),
            "base_sharpe":  bm.get("sharpe_ratio"),
            # error
            "error":        entry.get("error"),
        })
    return pd.DataFrame(rows)


def _load_best(artefacts_dir: str) -> Optional[dict]:
    path = Path(artefacts_dir) / "tv_research" / "tv_research_best.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_VERDICT_COLOR = {
    "HARD_REJECTED": "#EF5350",
    "REVISE":        "#FFA726",
    "APPROVED":      "#66BB6A",
    "UNSTABLE":      "#AB47BC",
    "ERROR":         "#9E9E9E",
}

_STATUS_COLOR = {
    "COMPLETED":          "#66BB6A",
    "TRANSLATION_FAILED": "#EF5350",
    "CODE_ERROR":         "#FF7043",
    "PIPELINE_FAILED":    "#EF5350",
    "DRY_RUN":            "#42A5F5",
    "STARTED":            "#9E9E9E",
}

_CATEGORY_COLOR = {
    "trend":       "#42A5F5",
    "momentum":    "#FF9800",
    "volatility":  "#AB47BC",
    "volume":      "#26A69A",
    "oscillator":  "#EC407A",
    "unknown":     "#9E9E9E",
}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _scatter_return_sharpe(df: pd.DataFrame) -> go.Figure:
    completed = df[df["status"] == "COMPLETED"].dropna(subset=["test_return", "test_sharpe"])
    if completed.empty:
        return go.Figure().update_layout(title="No completed runs yet", template="plotly_dark")

    fig = go.Figure()
    for cat, grp in completed.groupby("category"):
        color = _CATEGORY_COLOR.get(cat, "#9E9E9E")
        labels = [
            f"{r['name']}<br>Cat: {r['category']}<br>"
            f"Return: {r['test_return']*100:.1f}%  Sharpe: {r['test_sharpe']:.2f}<br>"
            f"DD: {(r['test_dd'] or 0)*100:.1f}%  Verdict: {r['verdict']}"
            for _, r in grp.iterrows()
        ]
        sizes  = [16 if r["is_best"] else 10 for _, r in grp.iterrows()]
        border = ["gold" if r["is_best"] else "rgba(0,0,0,0)" for _, r in grp.iterrows()]
        fig.add_trace(go.Scatter(
            x=grp["test_return"] * 100,
            y=grp["test_sharpe"],
            mode="markers+text",
            marker=dict(
                size=sizes, color=color,
                line=dict(color=border, width=2),
            ),
            text=grp["name"].str[:10],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=labels,
            hoverinfo="text",
            name=cat,
        ))

    fig.add_vline(x=50, line_dash="dash", line_color="gold",  opacity=0.5,
                  annotation_text="50% target")
    fig.add_hline(y=1.5, line_dash="dash", line_color="cyan", opacity=0.5,
                  annotation_text="Sharpe 1.5")
    fig.add_vline(x=20, line_dash="dot",  line_color="gray",  opacity=0.4,
                  annotation_text="20% min")
    fig.add_hline(y=1.0, line_dash="dot",  line_color="gray", opacity=0.4,
                  annotation_text="Sharpe 1.0")

    fig.update_layout(
        title="TV Indicators — Test Return vs Sharpe (by category)",
        xaxis_title="Annualized Return (%)",
        yaxis_title="Sharpe Ratio",
        height=500,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _bar_ranked_return(df: pd.DataFrame) -> go.Figure:
    completed = df[df["status"] == "COMPLETED"].dropna(subset=["test_return"])
    if completed.empty:
        return go.Figure().update_layout(title="No completed runs", template="plotly_dark")

    df_s = completed.sort_values("test_return")
    colors = [
        "gold"    if r["is_best"] else
        "#66BB6A" if r["test_return"] >= 0.50 else
        "#42A5F5" if r["test_return"] >= 0.20 else
        "#EF5350"
        for _, r in df_s.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=df_s["test_return"] * 100,
        y=df_s["name"],
        orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in df_s["test_return"]],
        textposition="outside",
        hovertemplate="%{y}<br>Return: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="gold", opacity=0.6,
                  annotation_text="50% target")
    fig.add_vline(x=20, line_dash="dot",  line_color="gray", opacity=0.4,
                  annotation_text="20% min")
    fig.update_layout(
        title="TV Indicators — Test Return (ranked)",
        xaxis_title="Annualized Return (%)",
        height=max(350, len(df_s) * 28),
        template="plotly_dark",
        margin=dict(l=160),
    )
    return fig


def _timeline_chart(df: pd.DataFrame) -> go.Figure:
    completed = df[df["status"] == "COMPLETED"].dropna(subset=["test_return"]).copy()
    if completed.empty:
        return go.Figure().update_layout(title="No completed runs", template="plotly_dark")

    completed = completed.sort_values("id")
    baseline = completed["base_return"].iloc[0] if "base_return" in completed.columns else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=completed["id"],
        y=completed["test_return"] * 100,
        mode="lines+markers",
        marker=dict(
            size=10,
            color=[_VERDICT_COLOR.get(v, "#9E9E9E") for v in completed["verdict"]],
            line=dict(color="white", width=1),
        ),
        text=completed["name"],
        hovertemplate="%{text}<br>Return: %{y:.1f}%<extra></extra>",
        name="Return",
    ))

    if baseline is not None:
        fig.add_hline(y=baseline * 100, line_dash="dot", line_color="gray",
                      opacity=0.6, annotation_text="Baseline")
    fig.add_hline(y=50, line_dash="dash", line_color="gold", opacity=0.5,
                  annotation_text="50% target")

    fig.update_layout(
        title="Return Convergence Timeline",
        xaxis_title="Iteration",
        yaxis_title="Annualized Return (%)",
        height=380,
        template="plotly_dark",
    )
    return fig


def _status_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["status"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker_colors=[_STATUS_COLOR.get(s, "#9E9E9E") for s in counts.index],
        hole=0.4,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Run Status Distribution",
        height=320,
        template="plotly_dark",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_tv_research(artefacts_dir: str) -> None:
    st.header("TV Indicator Research Results")

    df = _load_log(artefacts_dir)
    if df is None or df.empty:
        st.info(
            "No TV research results yet. Run:\n\n"
            "```\ntv_research --max-ideas 5 --source seed --trials 50\n```"
        )
        return

    best = _load_best(artefacts_dir)

    # ---- Summary banner ----
    total     = len(df)
    completed = (df["status"] == "COMPLETED").sum()
    errors    = df["status"].isin(["TRANSLATION_FAILED", "CODE_ERROR", "PIPELINE_FAILED"]).sum()
    approved  = (df["verdict"] == "APPROVED").sum()
    best_ret  = df["test_return"].max() if "test_return" in df.columns else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Runs",    total)
    c2.metric("Completed",     completed)
    c3.metric("Errors",        errors)
    c4.metric("Approved",      approved)
    c5.metric("Best Return",
              f"{best_ret*100:.1f}%" if best_ret is not None else "N/A")

    # ---- Goal status ----
    if best and best.get("test_metrics"):
        tm = best["test_metrics"]
        ret_ok  = (tm.get("annualized_return", 0) >= 0.50)
        shp_ok  = (tm.get("sharpe_ratio", 0)       >= 1.50)
        dd_ok   = (tm.get("max_drawdown", -999)     >= -0.25)
        goal_ok = ret_ok and shp_ok and dd_ok
        if goal_ok:
            st.success(f"Goal MET by {best['name']}!")
        else:
            parts = []
            if not ret_ok: parts.append(f"Return {tm.get('annualized_return',0)*100:.1f}% < 50%")
            if not shp_ok: parts.append(f"Sharpe {tm.get('sharpe_ratio',0):.3f} < 1.5")
            if not dd_ok:  parts.append(f"DD {tm.get('max_drawdown',0)*100:.1f}% < -25%")
            st.warning("Goal not yet met: " + " | ".join(parts))

    st.divider()

    # ---- Best indicator highlight ----
    if best:
        tm = best.get("test_metrics", {})
        st.subheader(f"Best Indicator: {best['name']} ({best.get('category', '?')})")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Return (Test)",  f"{tm.get('annualized_return', 0)*100:.1f}%")
        mc2.metric("Sharpe (Test)",  f"{tm.get('sharpe_ratio', 0):.3f}")
        mc3.metric("Max DD (Test)",  f"{tm.get('max_drawdown', 0)*100:.1f}%")
        mc4.metric("PF (Test)",      f"{tm.get('profit_factor', 0):.3f}")
        mc5.metric("Trades",         int(tm.get("total_trades", 0) or 0))
        if best.get("delta_vs_baseline"):
            dv = best["delta_vs_baseline"]
            st.caption(
                f"vs baseline: return {'+' if dv.get('return_delta', 0) >= 0 else ''}"
                f"{dv.get('return_delta', 0)*100:.1f}%"
                f" | sharpe {'+' if dv.get('sharpe_delta', 0) >= 0 else ''}"
                f"{dv.get('sharpe_delta', 0):.3f}"
            )

    st.divider()

    # ---- Charts row 1 ----
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(_scatter_return_sharpe(df), use_container_width=True)
    with col_b:
        st.plotly_chart(_status_pie(df), use_container_width=True)

    # ---- Timeline ----
    st.plotly_chart(_timeline_chart(df), use_container_width=True)

    # ---- Ranked bar chart ----
    st.plotly_chart(_bar_ranked_return(df), use_container_width=True)

    # ---- Full log table ----
    with st.expander(f"All {total} Runs", expanded=True):
        completed_df = df[df["status"] == "COMPLETED"].dropna(subset=["test_return"]).copy()
        if not completed_df.empty:
            display = completed_df[[
                "name", "category", "source", "test_return", "test_sharpe",
                "test_dd", "test_pf", "test_trades", "test_wr",
                "delta_return", "delta_sharpe", "verdict", "duration_s",
            ]].sort_values("test_return", ascending=False).copy()

            for col in ["test_return", "delta_return"]:
                display[col] = (display[col] * 100).round(1).astype(str) + "%"
            for col in ["test_sharpe", "delta_sharpe", "test_pf"]:
                display[col] = display[col].round(3)
            if "test_dd" in display.columns:
                display["test_dd"] = (display["test_dd"] * 100).round(1).astype(str) + "%"
            if "test_wr" in display.columns:
                display["test_wr"] = (display["test_wr"] * 100).round(1).astype(str) + "%"
            display.columns = [
                "Indicator", "Category", "Source", "Return%", "Sharpe",
                "MaxDD%", "PF", "Trades", "WinRate%",
                "dReturn%", "dSharpe", "Verdict", "Duration(s)",
            ]
            st.dataframe(display.set_index("Indicator"), use_container_width=True)
        else:
            st.info("No completed runs yet.")

    # ---- Errors / failures ----
    failed = df[df["status"].isin(["TRANSLATION_FAILED", "CODE_ERROR", "PIPELINE_FAILED"])]
    if not failed.empty:
        with st.expander(f"Failures ({len(failed)})", expanded=False):
            for _, row in failed.iterrows():
                st.warning(f"**{row['name']}** [{row['status']}]: {str(row['error'])[:200]}")

    # ---- Indicator detail expander ----
    completed_names = df[df["status"] == "COMPLETED"]["name"].tolist()
    if completed_names:
        with st.expander("Indicator Detail", expanded=False):
            selected = st.selectbox("Select indicator:", completed_names)
            row = df[df["name"] == selected].iloc[0]
            tm  = (row.get("test_metrics") or {}) if hasattr(row, "get") else {}

            # Try to load the generated Python module
            tv_indicators_dir = (
                Path(__file__).parents[3] / "src" / "trade2" / "features" / "tv_indicators"
            )
            mod_path = tv_indicators_dir / f"{selected}.py"
            if mod_path.exists():
                st.caption(f"Module: {mod_path.name}")
                with st.expander("Generated Python Code", expanded=False):
                    st.code(mod_path.read_text(), language="python")
            else:
                st.info(f"Module file not found: {mod_path}")
