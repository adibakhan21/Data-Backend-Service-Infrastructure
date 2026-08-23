#!/usr/bin/env python3
"""Generate synthetic trades and load them into SQLite.

Streams in chunks so peak memory stays flat regardless of --rows: a 5M-row load
never holds more than --chunk-size rows in a DataFrame at once.

Examples:
    python scripts/generate_data.py --rows 100000
    python scripts/generate_data.py --rows 5000000 --chunk-size 500000
    python scripts/generate_data.py --rows 50000 --seed 7 --db data/alt.db
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetrack import db  # noqa: E402
from tradetrack.config import settings  # noqa: E402
from tradetrack.generator import GeneratorConfig, generate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=100_000, help="Total trades to generate.")
    parser.add_argument("--chunk-size", type=int, default=250_000, help="Rows generated and written per batch.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed; same seed gives identical data.")
    parser.add_argument("--accounts", type=int, default=200, help="Size of the account pool.")
    parser.add_argument("--db", type=Path, default=settings.db_path, help="Target SQLite file.")
    parser.add_argument("--reset", action="store_true", help="Delete the database file before loading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows <= 0 or args.chunk_size <= 0:
        print("error: --rows and --chunk-size must be positive", file=sys.stderr)
        return 2

    if args.reset and args.db.exists():
        args.db.unlink()
        for suffix in ("-wal", "-shm"):  # WAL sidecar files
            sidecar = args.db.with_name(args.db.name + suffix)
            sidecar.unlink(missing_ok=True)
        print(f"removed existing {args.db}")

    started = time.perf_counter()
    written = 0
    with db.connect(args.db) as conn:
        db.init_schema(conn)

        chunk_index = 0
        while written < args.rows:
            size = min(args.chunk_size, args.rows - written)
            # Each chunk gets its own seed and start time so chunks do not
            # repeat one another's data, while the whole run stays reproducible.
            frame, _ = generate(
                GeneratorConfig(
                    n_rows=size,
                    n_accounts=args.accounts,
                    seed=args.seed + chunk_index,
                    start=GeneratorConfig().start + timedelta(hours=chunk_index),
                )
            )
            # Globally unique ids: the generator restarts numbering per chunk.
            frame["trade_id"] = [f"T{written + i:010d}" for i in range(size)]

            db.insert_frame(conn, frame)
            written += size
            chunk_index += 1
            print(f"  chunk {chunk_index}: {written:,}/{args.rows:,} rows", flush=True)

        total = db.count_trades(conn)

    elapsed = time.perf_counter() - started
    size_mb = args.db.stat().st_size / 1024 / 1024
    print(
        f"\nwrote {written:,} rows in {elapsed:.2f}s "
        f"({written / elapsed:,.0f} rows/s)\n"
        f"table now holds {total:,} rows; {args.db} is {size_mb:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
