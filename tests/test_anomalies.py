"""Anomaly rules, scored against the generator's injected ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradetrack.anomalies import (
    detect_all,
    detect_failure_rate_anomalies,
    detect_latency_anomalies,
    detect_quantity_anomalies,
)
from tradetrack.config import Settings
from tradetrack.generator import GeneratorConfig, generate


class TestLatencyRule:
    def test_recovers_injected_spikes(self, frame, plan):
        found = {a.entity_id for a in detect_latency_anomalies(frame)}
        injected = set(plan.latency_trade_ids)
        recall = len(found & injected) / len(injected)
        precision = len(found & injected) / len(found)
        assert recall > 0.90, f"recall {recall:.3f}"
        assert precision > 0.90, f"precision {precision:.3f}"

    def test_absolute_floor_silences_a_healthy_dataset(self):
        """The false-positive guard. Every dataset has a p99, so a bare
        above-p99 rule always alerts on 1% of a perfectly healthy venue."""
        frame = _frame_with(latency=np.linspace(5.0, 25.0, 500))
        assert detect_latency_anomalies(frame) == []

    def test_reason_names_the_threshold(self, frame):
        anomaly = detect_latency_anomalies(frame)[0]
        assert "threshold" in anomaly.reason and str(int(anomaly.threshold)) in anomaly.reason

    def test_empty_input(self, empty_frame):
        assert detect_latency_anomalies(empty_frame) == []


class TestFailureRateRule:
    def test_recovers_injected_bad_accounts(self, frame, plan):
        flagged = {a.entity_id for a in detect_failure_rate_anomalies(frame, "account_id")}
        assert set(plan.failing_accounts) <= flagged, "missed an injected account"

    def test_small_samples_are_ignored(self):
        """One failure out of one order is a 100% failure rate and means
        nothing. Without the minimum-sample floor it is the top alert."""
        frame = _frame_with(status=["REJECTED"], account="TINY")
        assert detect_failure_rate_anomalies(frame, "account_id") == []

    def test_rejects_an_invalid_grouping(self, frame):
        with pytest.raises(ValueError, match="account_id"):
            detect_failure_rate_anomalies(frame, "price")

    def test_empty_input(self, empty_frame):
        assert detect_failure_rate_anomalies(empty_frame) == []


class TestQuantityRule:
    def test_recovers_injected_block_orders(self, frame, plan):
        found = {a.entity_id for a in detect_quantity_anomalies(frame)}
        injected = set(plan.oversized_trade_ids)
        recall = len(found & injected) / len(injected)
        precision = len(found & injected) / len(found)
        assert recall > 0.75, f"recall {recall:.3f}"
        assert precision > 0.75, f"precision {precision:.3f}"

    def test_log_space_beats_raw_space_on_precision(self):
        """The finding this rule exists to encode: on lognormal order sizes a
        raw-scale MAD rule flags the entire natural tail. This test fails if
        someone 'simplifies' the rule back to raw space."""
        frame, plan = generate(GeneratorConfig(n_rows=30_000, seed=99))
        injected = set(plan.oversized_trade_ids)

        log_found = {a.entity_id for a in detect_quantity_anomalies(frame)}
        log_precision = len(log_found & injected) / len(log_found)

        raw_found: set[str] = set()
        for _, group in frame.groupby("symbol", sort=False):
            quantities = group["quantity"].astype(float)
            median = quantities.median()
            mad = (quantities - median).abs().median()
            raw_found |= set(group.loc[quantities > median + 3.5 * mad * 1.4826, "trade_id"])
        raw_precision = len(raw_found & injected) / len(raw_found)

        assert log_precision > 5 * raw_precision, f"log {log_precision:.3f} vs raw {raw_precision:.3f}"

    def test_uniform_sizes_produce_no_alerts(self):
        """MAD is zero, so the threshold would collapse onto the median."""
        assert detect_quantity_anomalies(_frame_with(quantity=[100] * 200)) == []

    def test_threshold_is_configurable(self, frame):
        strict = detect_quantity_anomalies(frame, Settings(quantity_mad_threshold=6.0))
        loose = detect_quantity_anomalies(frame, Settings(quantity_mad_threshold=2.5))
        assert len(strict) < len(loose)

    def test_empty_input(self, empty_frame):
        assert detect_quantity_anomalies(empty_frame) == []


class TestDetectAll:
    def test_sorted_by_severity(self, frame):
        ranks = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        severities = [ranks[a.severity.value] for a in detect_all(frame)]
        assert severities == sorted(severities, reverse=True)

    def test_every_finding_is_self_explaining(self, frame):
        """The point of a rules-based detector: each output must carry enough
        to audit it without rerunning anything."""
        for anomaly in detect_all(frame):
            assert anomaly.reason and len(anomaly.reason) > 20
            assert anomaly.metric_name and anomaly.entity_id
            assert anomaly.scope in {"trade", "symbol", "account"}

    def test_empty_input(self, empty_frame):
        assert detect_all(empty_frame) == []


def _frame_with(latency=None, status=None, quantity=None, account="ACC1") -> pd.DataFrame:
    """Build a minimal valid frame, varying one column."""
    if latency is not None:
        n = len(latency)
    elif status is not None:
        n = len(status)
    elif quantity is not None:
        n = len(quantity)
    else:
        raise ValueError("supply one of latency/status/quantity")

    return pd.DataFrame({
        "trade_id": [f"T{i}" for i in range(n)],
        "timestamp": pd.date_range("2025-01-06", periods=n, freq="s", tz="UTC"),
        "symbol": ["AAPL"] * n,
        "side": ["BUY"] * n,
        "quantity": list(quantity) if quantity is not None else [100] * n,
        "price": [195.0] * n,
        "order_status": list(status) if status is not None else ["FILLED"] * n,
        "execution_latency_ms": list(latency) if latency is not None else [15.0] * n,
        "account_id": [account] * n,
    })
