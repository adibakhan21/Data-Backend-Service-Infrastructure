#!/usr/bin/env python3
"""Measure the algorithm choices this project makes, against their naive twin.

Read the distinction this script is built around:

  * **Complexity** is a property of an algorithm. O(n log k) is true on any
    machine, forever, and is derived, not measured.
  * **Runtime** is a property of one machine on one day. Every number this
    script prints is a measurement on whatever laptop ran it, and it will not
    reproduce exactly anywhere else.

Both matter, and conflating them is how "we made it 40x faster" ends up meaning
"we ran it on a quieter machine". Every result below therefore reports the
measured speedup *and* the complexity change that explains it - and where the
complexity is unchanged (the generator, the SQLite batch), it says so, because
that speedup is a constant-factor win and nothing more.

Method: each case runs `--repeat` times; the **minimum** is reported, not the
mean. The minimum is the run least contaminated by scheduler noise, GC pauses
and other processes - the mean measures your machine's background load as much
as the code.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --scale 500000 --repeat 5 --output benchmarks/results.md
"""

from __future__ import annotations

import argparse
import platform
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetrack import db  # noqa: E402
from tradetrack.algorithms import (  # noqa: E402
    build_index,
    linear_lookup,
    naive_top_k,
    naive_window_failure_rate,
    sliding_window_failure_rate,
    top_k,
)
from tradetrack.generator import GeneratorConfig, generate, generate_rows_naive  # noqa: E402


def _fmt(seconds: float) -> str:
    """Format a duration in whichever unit keeps it readable.

    A per-operation cost in this project spans nanoseconds (a dict hit) to
    seconds (20k row-at-a-time commits). Forcing all of it into milliseconds
    prints `0.000` for the fastest case, which reads as "not measured" and
    makes any ratio derived from it look invented.
    """
    if seconds >= 1:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.3f} us"
    return f"{seconds * 1e9:.1f} ns"


@dataclass
class Result:
    """One benchmark comparison."""

    name: str
    n: int
    baseline_label: str
    optimised_label: str
    baseline_s: float
    optimised_s: float
    complexity_note: str

    @property
    def speedup(self) -> float:
        return self.baseline_s / self.optimised_s if self.optimised_s > 0 else float("inf")


def timed(fn: Callable[[], object], repeat: int) -> float:
    """Return the fastest of `repeat` runs, in seconds."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def bench_lookup(frame, repeat: int) -> Result:
    """Hash index vs linear scan, worst case (the key is the final row)."""
    records = frame.to_dict("records")
    target = records[-1]["trade_id"]
    index = build_index(records, key=lambda r: r["trade_id"])

    # A single dict.get takes tens of nanoseconds - far below the resolution of
    # perf_counter. Timing one would print 0.000 and yield a meaningless ratio,
    # so both sides are run in a loop and divided by the iteration count. The
    # dict side needs many more iterations than the scan side to clear the
    # timer floor, hence the asymmetric counts.
    scan_iterations = 10
    hash_iterations = 200_000

    return Result(
        name="Trade lookup by id (worst case: last row)",
        n=len(records),
        baseline_label="linear scan",
        optimised_label="hash index",
        baseline_s=timed(
            lambda: [linear_lookup(records, lambda r: r["trade_id"], target) for _ in range(scan_iterations)],
            repeat,
        ) / scan_iterations,
        optimised_s=timed(
            lambda: [index.get(target) for _ in range(hash_iterations)], repeat
        ) / hash_iterations,
        complexity_note="O(n) -> O(1) average. Index build is a one-time O(n).",
    )


def bench_top_k(frame, repeat: int, k: int = 10) -> Result:
    """Heap top-K vs sorting everything."""
    rows = frame[["symbol", "quantity"]].to_dict("records")
    key = lambda r: r["quantity"]  # noqa: E731
    return Result(
        name=f"Top-{k} by quantity",
        n=len(rows),
        baseline_label="full sort",
        optimised_label="size-k heap",
        baseline_s=timed(lambda: naive_top_k(rows, key, k), repeat),
        optimised_s=timed(lambda: top_k(rows, key, k), repeat),
        complexity_note=f"O(n log n) -> O(n log k) time; O(n) -> O(k) space at k={k}.",
    )


def bench_sliding_window(frame, repeat: int, window: int = 500) -> Result:
    """Running-sum window vs recomputing the window each step."""
    flags = frame["order_status"].isin({"CANCELLED", "REJECTED"}).tolist()
    # The naive version is O(n*w); on the full scale it would run for minutes,
    # so it is measured on a slice and the comparison is stated at that size.
    slice_size = min(len(flags), 50_000)
    subset = flags[:slice_size]
    return Result(
        name=f"Rolling failure rate (window={window})",
        n=slice_size,
        baseline_label="recompute window",
        optimised_label="running sum",
        baseline_s=timed(lambda: naive_window_failure_rate(subset, window), repeat),
        optimised_s=timed(lambda: sliding_window_failure_rate(subset, window), repeat),
        complexity_note=f"O(n*w) -> O(n) time at w={window}; O(n) -> O(w) space.",
    )


def bench_generation(rows: int, repeat: int) -> Result:
    """Vectorised numpy vs a per-row Python loop."""
    scale = min(rows, 100_000)  # the naive loop is slow enough to bound here
    cfg = GeneratorConfig(n_rows=scale)
    return Result(
        name="Synthetic data generation",
        n=scale,
        baseline_label="row-by-row Python",
        optimised_label="vectorised numpy",
        baseline_s=timed(lambda: generate_rows_naive(cfg), max(1, repeat // 3)),
        optimised_s=timed(lambda: generate(cfg), repeat),
        complexity_note="Both O(n). Constant-factor only: the per-row work moves from the Python interpreter into C.",
    )


def bench_sqlite_insert(frame, repeat: int) -> Result:
    """One transaction vs one commit per row."""
    scale = min(len(frame), 20_000)  # per-row commits fsync; keep this bounded
    subset = frame.head(scale)

    def run(insert_fn) -> Callable[[], None]:
        def inner() -> None:
            path = Path(tempfile.mkdtemp()) / "bench.db"
            conn = sqlite3.connect(str(path))
            try:
                conn.executescript(db.SCHEMA)
                insert_fn(conn, subset)
            finally:
                conn.close()
                path.unlink(missing_ok=True)

        return inner

    return Result(
        name="SQLite bulk insert",
        n=scale,
        baseline_label="commit per row",
        optimised_label="executemany, one txn",
        baseline_s=timed(run(db.insert_rows_one_by_one), 1),
        optimised_s=timed(run(db.insert_frame), repeat),
        complexity_note="Both O(n log n) for index maintenance. The win is fsync count: n commits -> 1.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=int, default=200_000, help="Rows for the scan-based benchmarks.")
    parser.add_argument("--repeat", type=int, default=3, help="Runs per case; the fastest is reported.")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.md"))
    args = parser.parse_args()

    print(f"generating {args.scale:,} rows...", flush=True)
    frame, _ = generate(GeneratorConfig(n_rows=args.scale))

    results: list[Result] = []
    for label, fn in [
        ("lookup", lambda: bench_lookup(frame, args.repeat)),
        ("top-k", lambda: bench_top_k(frame, args.repeat)),
        ("sliding window", lambda: bench_sliding_window(frame, args.repeat)),
        ("generation", lambda: bench_generation(args.scale, args.repeat)),
        ("sqlite insert", lambda: bench_sqlite_insert(frame, args.repeat)),
    ]:
        print(f"  running {label}...", flush=True)
        results.append(fn())

    machine = (
        f"{platform.machine()} / {platform.system()} {platform.release()} / "
        f"Python {platform.python_version()}"
    )

    lines = [
        "# Benchmark results",
        "",
        f"Machine: `{machine}`  ",
        f"Method: fastest of {args.repeat} runs per case (minimum, not mean - see the",
        "script docstring for why).",
        "",
        "**These runtimes are measurements from one machine on one day. The complexity",
        "column is the part that transfers.** Regenerate with `python scripts/run_benchmarks.py`.",
        "",
        "| Benchmark | n | Baseline | Optimised | Baseline | Optimised | Speedup | Complexity change |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.n:,} | {r.baseline_label} | {r.optimised_label} | "
            f"{_fmt(r.baseline_s)} | {_fmt(r.optimised_s)} | "
            f"{r.speedup:,.1f}x | {r.complexity_note} |"
        )
    lines += [
        "",
        "## Reading these honestly",
        "",
        "* **Lookup** and **rolling failure rate** are genuine complexity wins: the speedup",
        "  grows with n, so it would be larger on a bigger dataset and smaller on a smaller one.",
        "  The lookup ratio is large because it is O(n) vs O(1) at n=200,000 - quoting it as a",
        "  bare multiplier is meaningless without that n. The useful statement is the absolute",
        "  one: a full scan of 200,000 records costs milliseconds, a dict hit costs nanoseconds.",
        "* **Top-K** at 12 symbols would show nothing; it is benchmarked over all trades to make",
        "  the effect visible. In the API the input is the symbol count, where the sort would be fine.",
        "  The heap is there for the case where the group count is large, not for today's 12.",
        "* **Generation** and **SQLite insert** are constant-factor wins, not complexity wins.",
        "  Real, useful, and they would not change the shape of the scaling curve.",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print("\n" + "\n".join(lines[10:16 + len(results)]))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
