"""The in-memory snapshot and its SQLite round-trip."""

from __future__ import annotations

import pandas as pd

from tradetrack import db
from tradetrack.store import AnalyticsStore


class TestStore:
    def test_lookup_matches_the_frame(self, store, frame):
        target = frame["trade_id"].iloc[42]
        assert store.get_trade(target)["trade_id"] == target

    def test_unknown_id_returns_none(self, store):
        assert store.get_trade("NOT_A_TRADE") is None

    def test_index_covers_every_row(self, store, frame):
        assert len(store.trade_index) == len(frame)

    def test_frame_is_time_ordered(self, store):
        """Signals slice by position, so the snapshot must be sorted once at
        build time rather than re-sorted per request."""
        assert store.frame["timestamp"].is_monotonic_increasing

    def test_derived_views_are_populated(self, store):
        assert store.metrics is not None
        assert store.symbol_metrics and store.anomalies and store.signals
        assert store.build_seconds > 0

    def test_empty_store_is_flagged_not_broken(self, empty_frame):
        store = AnalyticsStore.build(empty_frame)
        assert store.is_empty
        assert store.get_trade("anything") is None
        assert store.metrics.total_trades == 0


class TestSqliteRoundTrip:
    def test_write_then_read_preserves_the_data(self, frame, tmp_path):
        path = tmp_path / "roundtrip.db"
        with db.connect(path) as conn:
            db.init_schema(conn)
            written = db.insert_frame(conn, frame)
            restored = db.load_frame(conn)

        assert written == len(frame) == len(restored)
        assert set(restored["trade_id"]) == set(frame["trade_id"])
        assert pd.api.types.is_datetime64_any_dtype(restored["timestamp"])

    def test_reloading_the_same_batch_is_idempotent(self, frame, tmp_path):
        """INSERT OR REPLACE on the primary key: a retried load must not
        duplicate rows. This is what makes a failed load safe to just re-run."""
        path = tmp_path / "idempotent.db"
        with db.connect(path) as conn:
            db.init_schema(conn)
            db.insert_frame(conn, frame)
            db.insert_frame(conn, frame)
            assert db.count_trades(conn) == len(frame)

    def test_init_schema_can_run_twice(self, tmp_path):
        with db.connect(tmp_path / "twice.db") as conn:
            db.init_schema(conn)
            db.init_schema(conn)

    def test_empty_insert_is_a_no_op(self, empty_frame, tmp_path):
        with db.connect(tmp_path / "empty.db") as conn:
            db.init_schema(conn)
            assert db.insert_frame(conn, empty_frame) == 0

    def test_missing_file_yields_an_empty_store(self, tmp_path):
        assert AnalyticsStore.from_sqlite(tmp_path / "does_not_exist.db").is_empty
