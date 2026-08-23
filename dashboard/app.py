"""Streamlit dashboard.

Reads the API over HTTP rather than importing the analytics package directly.
That is deliberate: it forces the API to be genuinely sufficient for a client,
so a second consumer (a notebook, a scheduled report, another service) needs no
new code. Importing the package here would have been less work and would have
let the HTTP surface quietly rot.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("TRADETRACK_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 10

st.set_page_config(page_title="TradeTrack Analytics", layout="wide")


@st.cache_data(ttl=30)
def fetch(path: str):
    """GET a path from the API, returning None on any failure.

    Cached for 30s so moving a filter does not re-hit the API for data that has
    not changed. Errors are surfaced in the UI rather than raised, because a
    dashboard that shows a stack trace is a dashboard nobody trusts.
    """
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"`GET {path}` failed: {exc}")
        return None


st.title("TradeTrack Analytics Engine")
st.caption(
    f"Reading from `{API_BASE}`. Synthetic data. Signals are descriptive rules, "
    "not predictions - see the README."
)

health = fetch("/health")
if health is None:
    st.warning(
        "Cannot reach the API. Start it with:\n\n"
        "```\nuvicorn tradetrack.api.main:app --app-dir src\n```"
    )
    st.stop()

if health["status"] != "ok":
    st.warning(
        f"API is **{health['status']}** with {health['trades_loaded']:,} trades loaded. "
        "Run `python scripts/generate_data.py` and restart the API."
    )
    st.stop()

metrics = fetch("/metrics")
anomalies = fetch("/anomalies?limit=200")
signals = fetch("/signals?limit=50")
top_symbols = fetch("/metrics/top?sort_by=notional&k=8")

if not all([metrics, anomalies, signals, top_symbols]):
    st.stop()

# ---- headline numbers ----------------------------------------------------
row = st.columns(5)
row[0].metric("Total trades", f"{metrics['total_trades']:,}")
row[1].metric("Order failure rate", f"{metrics['failure_rate']:.2%}")
row[2].metric("Avg latency", f"{metrics['avg_latency_ms']:.1f} ms")
row[3].metric("p95 latency", f"{metrics['p95_latency_ms']:.1f} ms")
row[4].metric("Anomalies", f"{anomalies['total']:,}")

st.caption(
    f"Window {metrics['window_start']} to {metrics['window_end']} - "
    f"{metrics['distinct_symbols']} symbols, {metrics['distinct_accounts']} accounts, "
    f"${metrics['total_notional']:,.0f} notional. "
    f"Snapshot built in {health['snapshot_build_seconds']:.2f}s."
)

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Top symbols by notional")
    top_frame = pd.DataFrame(top_symbols)
    st.bar_chart(top_frame.set_index("symbol")["total_notional"], height=260)
    st.dataframe(
        top_frame[["symbol", "trade_count", "failure_rate", "vwap", "p95_latency_ms"]],
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Signal summary")
    signal_frame = pd.DataFrame(signals["items"])
    if signal_frame.empty:
        st.info("No symbol has enough history for a signal yet.")
    else:
        st.bar_chart(signal_frame["action"].value_counts(), height=200)
        st.dataframe(
            signal_frame[["symbol", "action", "confidence", "momentum_pct", "volume_ratio"]],
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.subheader("Anomalies")

anomaly_frame = pd.DataFrame(anomalies["items"])
if anomaly_frame.empty:
    st.success("No anomalies detected in this snapshot.")
else:
    counts = st.columns(2)
    counts[0].bar_chart(anomaly_frame["anomaly_type"].value_counts(), height=220)
    counts[1].bar_chart(anomaly_frame["severity"].value_counts(), height=220)

    available = sorted(anomaly_frame["anomaly_type"].unique())
    chosen = st.multiselect("Filter by type", available, default=available)
    filtered = anomaly_frame[anomaly_frame["anomaly_type"].isin(chosen)]

    st.caption(
        f"Showing {len(filtered)} of {anomalies['total']} findings. Every row states the "
        "metric, the threshold it crossed, and why - that is the point of using rules "
        "rather than a model."
    )
    st.dataframe(
        filtered[["severity", "anomaly_type", "scope", "entity_id", "symbol", "reason"]],
        use_container_width=True,
        hide_index=True,
    )
