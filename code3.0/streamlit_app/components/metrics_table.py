"""
components/metrics_table.py - Metrics display and comparison.
"""
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd


_KEY_METRICS = [
    ("annualized_return", "Annual Return", "{:.1%}"),
    ("sharpe_ratio",      "Sharpe Ratio",  "{:.3f}"),
    ("max_drawdown",      "Max Drawdown",  "{:.1%}"),
    ("total_trades",      "Total Trades",  "{:.0f}"),
    ("profit_factor",     "Profit Factor", "{:.2f}"),
    ("win_rate",          "Win Rate",      "{:.1%}"),
]


def render_metric_cards(metrics: Dict[str, Any], comparison: Optional[Dict[str, Any]] = None) -> None:
    """
    Render metric cards row. If comparison provided, shows deltas.
    """
    cols = st.columns(len(_KEY_METRICS))
    for col, (key, label, fmt) in zip(cols, _KEY_METRICS):
        val = metrics.get(key, None)
        if val is None:
            col.metric(label, "N/A")
            continue

        val_str = fmt.format(val)

        if comparison is not None:
            cval = comparison.get(key, None)
            if cval is not None:
                delta = val - cval
                delta_str = fmt.format(delta)
                col.metric(label, val_str, delta=delta_str)
                continue

        col.metric(label, val_str)


def render_full_metrics_table(metrics: Dict[str, Any]) -> None:
    """Render full metrics as a DataFrame table."""
    rows = []
    for k, v in metrics.items():
        if isinstance(v, float):
            rows.append({"Metric": k, "Value": f"{v:.4f}"})
        else:
            rows.append({"Metric": k, "Value": str(v)})

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)
