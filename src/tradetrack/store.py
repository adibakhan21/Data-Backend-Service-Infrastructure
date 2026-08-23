"""In-memory analytics store: the object every API request reads from.

Why a store at all
------------------
Recomputing p95 latency or the anomaly set per request would make every
endpoint O(n log n) on the whole dataset. Instead the expensive work happens
once at startup and each request becomes a slice or a dict lookup.

That is a cache, so it has a cache's tradeoff, stated plainly: the store is a
point-in-time snapshot. New rows written to SQLite are invisible until
`reload()` is called. For a batch-analytics service that is the correct
tradeoff; `docs/system_design.md` covers what changes when the data becomes a
live stream.

The `trade_index` is the concrete payoff: `GET /trades/{id}` is an O(1) average
dict hit rather than an O(n) scan of a DataFrame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import db
from .anomalies import detect_all
from .config import Settings, settings as default_settings
from .models import Anomaly, GlobalMetrics, Signal, SymbolMetrics
from .processing import aggregate_by_symbol_pandas, global_metrics
from .signals import generate_signals


@dataclass
class AnalyticsStore:
    """Precomputed analytics over one snapshot of the trades table.

    Attributes:
        frame: Every trade, sorted by timestamp.
        trade_index: trade_id -> positional row index. O(1) average lookup.
        metrics: Dataset-wide summary.
        symbol_metrics: Per-symbol aggregates.
        anomalies: All findings, severity-sorted.
        signals: One per eligible symbol, confidence-sorted.
        loaded_at: When this snapshot was built (UTC).
        build_seconds: Wall-clock cost of building it, surfaced on /health so
            the cost of the snapshot is observable rather than folklore.
    """

    frame: pd.DataFrame
    trade_index: dict[str, int] = field(default_factory=dict)
    metrics: GlobalMetrics | None = None
    symbol_metrics: dict[str, SymbolMetrics] = field(default_factory=dict)
    anomalies: list[Anomaly] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    loaded_at: datetime | None = None
    build_seconds: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.frame.empty

    @classmethod
    def build(cls, frame: pd.DataFrame, config: Settings | None = None) -> "AnalyticsStore":
        """Compute every derived view once.

        Complexity:
            Time  O(n log n) - the sort and the percentiles dominate.
            Space O(n)
        """
        started = time.perf_counter()
        cfg = config or default_settings

        if frame.empty:
            return cls(
                frame=frame,
                metrics=global_metrics(frame),
                loaded_at=datetime.now(timezone.utc),
                build_seconds=time.perf_counter() - started,
            )

        ordered = frame.sort_values("timestamp").reset_index(drop=True)
        return cls(
            frame=ordered,
            # Positional index, not the row itself: storing row objects would
            # duplicate the whole dataset in memory for no gain.
            trade_index={tid: i for i, tid in enumerate(ordered["trade_id"])},
            metrics=global_metrics(ordered),
            symbol_metrics=aggregate_by_symbol_pandas(ordered),
            anomalies=detect_all(ordered, cfg),
            signals=generate_signals(ordered, cfg),
            loaded_at=datetime.now(timezone.utc),
            build_seconds=round(time.perf_counter() - started, 4),
        )

    @classmethod
    def from_sqlite(cls, db_path: Path | str, config: Settings | None = None) -> "AnalyticsStore":
        """Load from SQLite and build. Returns an empty store if the file is absent."""
        path = Path(db_path)
        if not path.exists():
            return cls.build(pd.DataFrame(columns=db.COLUMNS), config)
        with db.connect(path) as conn:
            db.init_schema(conn)
            return cls.build(db.load_frame(conn), config)

    def get_trade(self, trade_id: str) -> dict | None:
        """Fetch one trade by id.

        Complexity:
            Time O(1) average - one dict lookup, then one positional row access.
            This is the endpoint the `linear_lookup` benchmark exists to
            contrast against.
        """
        position = self.trade_index.get(trade_id)
        if position is None:
            return None
        return self.frame.iloc[position].to_dict()
