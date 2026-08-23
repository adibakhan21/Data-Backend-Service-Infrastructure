"""SQLite persistence.

SQLite and not PostgreSQL, deliberately. This workload is read-mostly and
single-writer: one batch load, then analytical reads. SQLite handles that
without a server process, a connection pool, or a docker-compose service, which
means `git clone && make run` actually works. `docs/system_design.md` sets out
exactly when that stops being true and Postgres becomes the right answer.

Writes go through `executemany` inside a single transaction. Row-at-a-time
`INSERT` with autocommit costs one fsync per row; batching a million rows into
one transaction is roughly two orders of magnitude faster, and the benchmark
in `scripts/run_benchmarks.py` measures the gap rather than asserting it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id             TEXT PRIMARY KEY,
    timestamp            TEXT NOT NULL,
    symbol               TEXT NOT NULL,
    side                 TEXT NOT NULL,
    quantity             INTEGER NOT NULL CHECK (quantity > 0),
    price                REAL NOT NULL CHECK (price > 0),
    order_status         TEXT NOT NULL,
    execution_latency_ms REAL NOT NULL CHECK (execution_latency_ms >= 0),
    account_id           TEXT NOT NULL
);

-- Covers `GET /trades?symbol=...`, the most common filter. Without it SQLite
-- full-scans the table for every symbol query.
CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_account   ON trades (account_id);
CREATE INDEX IF NOT EXISTS idx_trades_status    ON trades (order_status);
-- Range scans for time-window queries, and the natural ordering for `recent`.
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp);
"""

INSERT_SQL = """
INSERT OR REPLACE INTO trades
    (trade_id, timestamp, symbol, side, quantity, price,
     order_status, execution_latency_ms, account_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

COLUMNS = [
    "trade_id", "timestamp", "symbol", "side", "quantity",
    "price", "order_status", "execution_latency_ms", "account_id",
]


@contextmanager
def connect(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """Open a connection with sane pragmas, and always close it.

    WAL lets readers proceed while a write is in flight, which is what keeps
    the API responsive during a bulk load. `synchronous=NORMAL` trades a
    fsync-per-commit for speed; with WAL the durability loss is bounded to the
    most recent transaction on power loss, which is acceptable for a dataset we
    can simply regenerate.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not already exist (idempotent)."""
    conn.executescript(SCHEMA)
    conn.commit()


def insert_frame(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    """Bulk-insert a frame of trades. Returns the number of rows written.

    `INSERT OR REPLACE` on the `trade_id` primary key makes loading idempotent:
    re-running the loader with the same data yields the same table rather than
    duplicating it, so a retried or half-finished load is always safe to repeat.

    Complexity:
        Time  O(n log n) - each insert maintains four B-tree indexes.
        Space O(1) beyond the batch being written.
    """
    if frame.empty:
        return 0

    payload = frame[COLUMNS].copy()
    payload["timestamp"] = payload["timestamp"].astype(str)
    rows = list(payload.itertuples(index=False, name=None))

    with conn:  # single transaction: one fsync for the whole batch
        conn.executemany(INSERT_SQL, rows)
    return len(rows)


def insert_rows_one_by_one(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    """Autocommit row-at-a-time insert - the control arm for the benchmark."""
    payload = frame[COLUMNS].copy()
    payload["timestamp"] = payload["timestamp"].astype(str)
    count = 0
    for row in payload.itertuples(index=False, name=None):
        conn.execute(INSERT_SQL, row)
        conn.commit()  # the expensive part: one fsync per row
        count += 1
    return count


def load_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read every trade back into a DataFrame.

    Deliberately loads the whole table: the API serves analytics over the full
    dataset, and at the scales this project targets (<= a few million rows) an
    in-process frame is far faster than round-tripping SQL per request. The
    point at which this stops being true, and what replaces it, is the first
    bottleneck discussed in `docs/system_design.md`.
    """
    frame = pd.read_sql_query("SELECT * FROM trades", conn)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="ISO8601")
    return frame


def count_trades(conn: sqlite3.Connection) -> int:
    """Number of rows in the trades table."""
    return int(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0])
