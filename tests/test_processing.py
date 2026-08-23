"""Validation and aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from tradetrack.processing import (
    aggregate_by_symbol,
    aggregate_by_symbol_pandas,
    global_metrics,
    is_failure,
    validate_records,
)

VALID_RECORD = {
    "trade_id": "T1",
    "timestamp": "2025-01-06T09:30:00Z",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 100,
    "price": 195.0,
    "order_status": "FILLED",
    "execution_latency_ms": 12.5,
    "account_id": "ACC1",
}


class TestIsFailure:
    @pytest.mark.parametrize("status,expected", [
        ("FILLED", False), ("PARTIAL", False), ("CANCELLED", True), ("REJECTED", True),
    ])
    def test_classification(self, status, expected):
        assert is_failure(status) is expected


class TestValidation:
    def test_accepts_a_good_record(self):
        report = validate_records([VALID_RECORD])
        assert len(report.valid) == 1 and not report.rejected

    def test_normalises_symbol_case(self):
        report = validate_records([{**VALID_RECORD, "symbol": "aapl"}])
        assert report.valid[0].symbol == "AAPL"

    @pytest.mark.parametrize("field,bad_value", [
        ("quantity", 0), ("quantity", -5), ("price", 0), ("price", -1.0),
        ("execution_latency_ms", -0.1), ("order_status", "PENDING"),
        ("side", "HOLD"), ("trade_id", ""), ("account_id", ""),
    ])
    def test_rejects_invalid_values(self, field, bad_value):
        report = validate_records([{**VALID_RECORD, field: bad_value}])
        assert not report.valid
        assert len(report.rejected) == 1
        assert field in report.rejected[0][1]

    def test_missing_field_is_rejected(self):
        record = {k: v for k, v in VALID_RECORD.items() if k != "price"}
        assert validate_records([record]).rejection_rate == 1.0

    def test_one_bad_row_does_not_abort_the_batch(self):
        """The property that matters at ingest: a malformed record must not
        take the other 999,999 down with it."""
        report = validate_records([VALID_RECORD, {**VALID_RECORD, "trade_id": "T2", "price": -1}, VALID_RECORD])
        assert len(report.valid) == 2 and len(report.rejected) == 1
        assert report.total == 3 and report.rejection_rate == pytest.approx(1 / 3)

    def test_empty_batch(self):
        report = validate_records([])
        assert report.total == 0 and report.rejection_rate == 0.0


class TestAggregation:
    def test_both_paths_agree_exactly(self, frame):
        """The pure-Python reference and the pandas fast path must not drift."""
        reference = aggregate_by_symbol(validate_records(frame.to_dict("records")).valid)
        fast = aggregate_by_symbol_pandas(frame)
        assert {k: v.model_dump() for k, v in reference.items()} == {k: v.model_dump() for k, v in fast.items()}

    def test_counts_sum_to_the_dataset(self, frame):
        metrics = aggregate_by_symbol_pandas(frame)
        assert sum(m.trade_count for m in metrics.values()) == len(frame)

    def test_failure_rate_is_a_proportion(self, frame):
        for m in aggregate_by_symbol_pandas(frame).values():
            assert 0.0 <= m.failure_rate <= 1.0
            assert m.failed_count <= m.trade_count

    def test_vwap_is_hand_checkable(self):
        """Two trades: 100 @ 10 and 300 @ 20 -> VWAP 17.5, not the mean of 15."""
        frame = pd.DataFrame([
            {**VALID_RECORD, "trade_id": "A", "quantity": 100, "price": 10.0},
            {**VALID_RECORD, "trade_id": "B", "quantity": 300, "price": 20.0},
        ])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        assert aggregate_by_symbol_pandas(frame)["AAPL"].vwap == pytest.approx(17.5)

    def test_empty_input(self, empty_frame):
        assert aggregate_by_symbol_pandas(empty_frame) == {}
        assert aggregate_by_symbol([]) == {}


class TestGlobalMetrics:
    def test_matches_manual_counts(self, frame):
        metrics = global_metrics(frame)
        assert metrics.total_trades == len(frame)
        assert metrics.distinct_symbols == frame["symbol"].nunique()
        assert metrics.failed_trades == int(frame["order_status"].isin({"CANCELLED", "REJECTED"}).sum())

    def test_percentiles_are_ordered(self, frame):
        m = global_metrics(frame)
        assert m.avg_latency_ms <= m.p95_latency_ms <= m.p99_latency_ms

    def test_empty_frame_returns_zeros_not_an_error(self, empty_frame):
        m = global_metrics(empty_frame)
        assert m.total_trades == 0 and m.failure_rate == 0.0 and m.window_start is None
