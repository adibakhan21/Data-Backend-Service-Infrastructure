"""Evidence-gathering tools available to the investigator.

Each tool is a thin, read-only wrapper over the analytics the engine already
computes. Three design rules, each of which exists because of how tool-calling
models fail:

1. **Every tool takes exactly one string argument.** Multi-argument tool calls
   are where small models most often produce malformed JSON; collapsing the
   signature removes that failure mode entirely.
2. **Every tool returns a short, factual string.** Returning a large object
   invites the model to summarise it badly. Returning a sentence it can quote
   keeps the reasoning grounded.
3. **A tool never raises.** An unknown symbol returns "no data", not an
   exception. A crashing tool ends the graph run; a tool that reports absence
   lets the agent reason about it.
"""

from __future__ import annotations

from collections.abc import Callable

from ..store import AnalyticsStore


def build_tools(store: AnalyticsStore) -> dict[str, Callable[[str], str]]:
    """Bind the tool implementations to one snapshot.

    Returns a name -> callable mapping. The names are what the model sees and
    what `evaluate_agents.py` scores tool selection against, so they are chosen
    to be self-describing rather than short.
    """

    def get_symbol_metrics(symbol: str) -> str:
        """Baseline statistics for one symbol - the comparison an anomaly needs."""
        metrics = store.symbol_metrics.get(symbol.strip().upper())
        if metrics is None:
            return f"No data for symbol '{symbol}'."
        return (
            f"{metrics.symbol}: {metrics.trade_count} trades, "
            f"failure rate {metrics.failure_rate:.2%}, VWAP {metrics.vwap:.2f}, "
            f"avg latency {metrics.avg_latency_ms:.1f}ms, p95 {metrics.p95_latency_ms:.1f}ms."
        )

    def get_account_history(account_id: str) -> str:
        """Order history for one account - distinguishes a bad actor from a bad day."""
        rows = store.frame[store.frame["account_id"] == account_id.strip()]
        if rows.empty:
            return f"No trades for account '{account_id}'."
        failures = int(rows["order_status"].isin({"CANCELLED", "REJECTED"}).sum())
        return (
            f"{account_id}: {len(rows)} orders across {rows['symbol'].nunique()} symbols, "
            f"{failures} failed ({failures / len(rows):.2%}), "
            f"median size {int(rows['quantity'].median())}, "
            f"avg latency {rows['execution_latency_ms'].mean():.1f}ms."
        )

    def get_trade_detail(trade_id: str) -> str:
        """The raw record behind a trade-scoped anomaly."""
        record = store.get_trade(trade_id.strip())
        if record is None:
            return f"No trade with id '{trade_id}'."
        return (
            f"{record['trade_id']}: {record['side']} {record['quantity']} "
            f"{record['symbol']} @ {record['price']}, status {record['order_status']}, "
            f"latency {record['execution_latency_ms']}ms, account {record['account_id']}."
        )

    def get_global_baseline(_: str = "") -> str:
        """Book-wide context. Takes an ignored argument so every tool has one
        signature - the model does not have to special-case a no-arg call."""
        metrics = store.metrics
        if metrics is None:
            return "No global metrics available."
        return (
            f"Book-wide: {metrics.total_trades} trades, failure rate "
            f"{metrics.failure_rate:.2%}, avg latency {metrics.avg_latency_ms:.1f}ms, "
            f"p95 {metrics.p95_latency_ms:.1f}ms, p99 {metrics.p99_latency_ms:.1f}ms."
        )

    return {
        "get_symbol_metrics": get_symbol_metrics,
        "get_account_history": get_account_history,
        "get_trade_detail": get_trade_detail,
        "get_global_baseline": get_global_baseline,
    }


# These strings are the only thing the model knows about each tool, so they are
# part of the program, not documentation.
#
# An earlier version described get_trade_detail as "use to inspect a trade-scoped
# anomaly", which directly contradicted the answer key in expected_first_tool.
# Measured tool-selection accuracy was 36.8%: the model was following the
# description correctly and being marked wrong for it. The descriptions below
# state what each tool ESTABLISHES rather than when to reach for it, and say
# explicitly that the trade's own fields are already in context - which is the
# actual reason re-fetching them establishes nothing.
TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_symbol_metrics": (
        "Baseline statistics for a ticker. Establishes what is NORMAL for that symbol, "
        "so an observed size or latency can be judged against it. Argument: the ticker."
    ),
    "get_account_history": (
        "Order history for one account. Establishes whether an account fails or trades "
        "unusually as a pattern rather than once. Argument: the account id."
    ),
    "get_trade_detail": (
        "The raw record for one trade. NOTE: the anomaly context above already contains "
        "this trade's symbol, size, status and latency, so this usually establishes "
        "nothing new. Use it only when you need a field the anomaly does not show, such "
        "as the side or the owning account. Argument: the trade id."
    ),
    "get_global_baseline": (
        "Book-wide statistics. Establishes whether the whole venue is degraded rather "
        "than one entity. Argument is ignored."
    ),
}


def expected_first_tool(anomaly_type: str, scope: str) -> str:
    """The tool a competent analyst would reach for first.

    This is the ground-truth rubric `evaluate_agents.py` scores against, so the
    reasoning behind each choice is worth stating:

    * An account-scoped failure-rate anomaly is *about* the account, so the
      account's history is the first thing to look at.
    * A trade-scoped anomaly needs its symbol's baseline before the trade means
      anything - "6,000 units" is unremarkable in one name and extraordinary in
      another. The trade's own detail is already embedded in the anomaly's
      reason string, so re-fetching it establishes nothing new.
    * A symbol-scoped failure-rate anomaly needs the book-wide rate to tell a
      bad symbol apart from a bad day for everyone.

    Judging only the FIRST call, not the whole sequence, is deliberate: later
    calls legitimately depend on what the first one returned, so scoring them
    would penalise correct adaptive behaviour.
    """
    if anomaly_type == "HIGH_FAILURE_RATE":
        return "get_account_history" if scope == "account" else "get_global_baseline"
    return "get_symbol_metrics"
