"""The processing pipeline: validate -> aggregate -> summarise.

Two aggregation paths exist on purpose:

* `aggregate_by_symbol` works on plain dicts using `algorithms.group_reduce`.
  It is the readable, dependency-free reference implementation and the one the
  complexity discussion in the docs refers to.
* `aggregate_by_symbol_pandas` does the same thing with a vectorised groupby.
  It is what the API actually calls on large datasets.

`tests/test_processing.py` asserts the two agree, so the fast path can never
silently drift from the reference.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .algorithms import group_reduce, percentile
from .models import FAILURE_STATUSES, GlobalMetrics, SymbolMetrics, Trade


def is_failure(order_status: str) -> bool:
    """A trade is a processing failure iff it was CANCELLED or REJECTED.

    Defined once, here, so every failure-rate number in the system - API,
    dashboard, anomaly rules, tests - uses the same denominator. Diverging
    definitions of "failure" is the single most common way these dashboards
    start disagreeing with each other.
    """
    return order_status in FAILURE_STATUSES


@dataclass
class ValidationReport:
    """Outcome of validating a batch of raw records.

    Attributes:
        valid: Records that passed every `models.Trade` constraint.
        rejected: (row_index, error_message) for each record that failed.
    """

    valid: list[Trade]
    rejected: list[tuple[int, str]]

    @property
    def total(self) -> int:
        return len(self.valid) + len(self.rejected)

    @property
    def rejection_rate(self) -> float:
        return len(self.rejected) / self.total if self.total else 0.0


def validate_records(records: Iterable[dict]) -> ValidationReport:
    """Validate raw dicts against the `Trade` model.

    Bad rows are collected rather than raised, so one malformed record cannot
    abort ingest of a million good ones - the caller decides what to do with
    the rejection rate.

    Complexity:
        Time  O(n) - one model construction per record.
        Space O(n)
    """
    valid: list[Trade] = []
    rejected: list[tuple[int, str]] = []
    for i, raw in enumerate(records):
        try:
            valid.append(Trade.model_validate(raw))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(p) for p in first["loc"]) or "<record>"
            rejected.append((i, f"{location}: {first['msg']}"))
    return ValidationReport(valid=valid, rejected=rejected)


# --------------------------------------------------------------------------
# Reference aggregation (pure Python, O(n))
# --------------------------------------------------------------------------


def _new_bucket() -> dict:
    return {
        "trade_count": 0,
        "failed_count": 0,
        "total_quantity": 0,
        "total_notional": 0.0,
        "latency_sum": 0.0,
        "latencies": [],
    }


def _reduce_trade(bucket: dict, trade: Trade) -> dict:
    """Fold one trade into its symbol's bucket. Every step is O(1)."""
    bucket["trade_count"] += 1
    bucket["failed_count"] += int(is_failure(trade.order_status.value))
    bucket["total_quantity"] += trade.quantity
    bucket["total_notional"] += trade.notional
    bucket["latency_sum"] += trade.execution_latency_ms
    bucket["latencies"].append(trade.execution_latency_ms)
    return bucket


def aggregate_by_symbol(trades: Sequence[Trade]) -> dict[str, SymbolMetrics]:
    """Reference per-symbol aggregation using a hash map.

    Complexity:
        Time  O(n + sum over symbols of m log m) where m is that symbol's trade
              count. The O(n) pass does the sums; the log factor comes only
              from sorting each symbol's latencies to get p95. If p95 were
              dropped, this would be a clean O(n).
        Space O(n) - because p95 needs every latency retained. That is the real
              memory cost of exact percentiles and the reason production
              systems reach for t-digest or HdrHistogram instead (see
              docs/system_design.md).
    """
    buckets = group_reduce(trades, key=lambda t: t.symbol, initial=_new_bucket, reducer=_reduce_trade)

    metrics: dict[str, SymbolMetrics] = {}
    for symbol, bucket in buckets.items():
        count = bucket["trade_count"]
        metrics[symbol] = SymbolMetrics(
            symbol=symbol,
            trade_count=count,
            failed_count=bucket["failed_count"],
            failure_rate=bucket["failed_count"] / count,
            total_quantity=bucket["total_quantity"],
            total_notional=round(bucket["total_notional"], 2),
            vwap=round(bucket["total_notional"] / bucket["total_quantity"], 4)
            if bucket["total_quantity"]
            else 0.0,
            avg_latency_ms=round(bucket["latency_sum"] / count, 3),
            p95_latency_ms=round(percentile(bucket["latencies"], 95), 3),
        )
    return metrics


# --------------------------------------------------------------------------
# Fast aggregation (pandas, same result)
# --------------------------------------------------------------------------


def aggregate_by_symbol_pandas(frame: pd.DataFrame) -> dict[str, SymbolMetrics]:
    """Vectorised equivalent of `aggregate_by_symbol`.

    Same O(n) work, but the per-row cost happens in C rather than in the
    Python interpreter. On 1M rows this is the difference between a request
    that returns and one that times out.
    """
    if frame.empty:
        return {}

    work = frame.copy()
    work["is_failure"] = work["order_status"].isin(FAILURE_STATUSES)
    work["notional"] = work["quantity"] * work["price"]

    grouped = work.groupby("symbol", sort=False).agg(
        trade_count=("trade_id", "size"),
        failed_count=("is_failure", "sum"),
        total_quantity=("quantity", "sum"),
        total_notional=("notional", "sum"),
        avg_latency_ms=("execution_latency_ms", "mean"),
        # np.percentile uses the same linear-interpolation method as
        # `algorithms.percentile`, so the two aggregation paths agree to
        # floating-point tolerance. They are not bit-identical: the two
        # implementations sum in different orders. `tests/test_processing.py`
        # asserts agreement within 1e-6, not exact equality, for that reason.
        p95_latency_ms=("execution_latency_ms", lambda s: float(np.percentile(s.to_numpy(), 95))),
    )

    metrics: dict[str, SymbolMetrics] = {}
    for symbol, row in grouped.iterrows():
        count = int(row["trade_count"])
        quantity = int(row["total_quantity"])
        notional = float(row["total_notional"])
        metrics[str(symbol)] = SymbolMetrics(
            symbol=str(symbol),
            trade_count=count,
            failed_count=int(row["failed_count"]),
            failure_rate=int(row["failed_count"]) / count,
            total_quantity=quantity,
            total_notional=round(notional, 2),
            vwap=round(notional / quantity, 4) if quantity else 0.0,
            avg_latency_ms=round(float(row["avg_latency_ms"]), 3),
            p95_latency_ms=round(float(row["p95_latency_ms"]), 3),
        )
    return metrics


def global_metrics(frame: pd.DataFrame) -> GlobalMetrics:
    """Dataset-wide summary.

    Complexity:
        Time  O(n log n) - the percentile sort dominates.
        Space O(n)
    """
    if frame.empty:
        return GlobalMetrics(
            total_trades=0, failed_trades=0, failure_rate=0.0,
            avg_latency_ms=0.0, p95_latency_ms=0.0, p99_latency_ms=0.0,
            total_notional=0.0, distinct_symbols=0, distinct_accounts=0,
        )

    failures = int(frame["order_status"].isin(FAILURE_STATUSES).sum())
    latency = frame["execution_latency_ms"]
    total = len(frame)
    return GlobalMetrics(
        total_trades=total,
        failed_trades=failures,
        failure_rate=round(failures / total, 6),
        avg_latency_ms=round(float(latency.mean()), 3),
        p95_latency_ms=round(float(latency.quantile(0.95)), 3),
        p99_latency_ms=round(float(latency.quantile(0.99)), 3),
        total_notional=round(float((frame["quantity"] * frame["price"]).sum()), 2),
        distinct_symbols=int(frame["symbol"].nunique()),
        distinct_accounts=int(frame["account_id"].nunique()),
        # `.floor("us")` first: datetime cannot hold nanoseconds, and the
        # implicit truncation otherwise emits a UserWarning on every call.
        window_start=frame["timestamp"].min().floor("us").to_pydatetime(),
        window_end=frame["timestamp"].max().floor("us").to_pydatetime(),
    )
