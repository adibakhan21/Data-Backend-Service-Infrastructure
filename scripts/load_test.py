#!/usr/bin/env python3
"""Measure API latency and throughput under concurrent load.

Until this existed, every latency number in the project was per-operation
(a function call, a benchmark loop). None of them said anything about what the
service does when several clients hit it at once, which is the question that
actually matters for an HTTP service. The README listed that as its largest
known gap; this closes it.

Self-contained: threads plus `requests`, no external load-testing binary. The
client is I/O-bound so the GIL is not the limiter at these concurrency levels,
but the honest caveat stands - **this measures client-observed latency from one
machine against a local server**, so it includes loopback and client overhead
and is a floor on real-world latency, not a prediction of it.

Reports the full distribution rather than a mean. A mean latency hides exactly
the tail that decides whether a service feels fast.

Usage:
    uvicorn tradetrack.api.main:app --app-dir src     # in another terminal
    python scripts/load_test.py
    python scripts/load_test.py --concurrency 32 --requests 2000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetrack.algorithms import percentile  # noqa: E402

# Chosen to span the cost profile: an O(1) attribute read, an O(n) filtered
# scan, an O(1) hash lookup, and a heap top-K. If they all came back identical
# it would mean the load generator, not the server, was the bottleneck.
ENDPOINTS = [
    ("/health", "liveness, no data access"),
    ("/metrics", "precomputed, O(1) read"),
    ("/trades?limit=100", "O(n) mask + serialise 100 rows"),
    ("/trades?symbol=NVDA&limit=50", "O(n) filtered mask"),
    ("/trades/T0000000042", "O(1) hash lookup"),
    ("/metrics/top?sort_by=notional&k=5", "heap top-K"),
    ("/anomalies?limit=50", "filter + slice"),
]


def hammer(base_url: str, path: str, n: int, concurrency: int) -> dict:
    """Fire `n` requests at `path` with `concurrency` workers."""
    session_pool = [requests.Session() for _ in range(concurrency)]
    latencies: list[float] = []
    errors = 0

    def one(i: int) -> tuple[float, bool]:
        session = session_pool[i % concurrency]
        start = time.perf_counter()
        try:
            response = session.get(f"{base_url}{path}", timeout=30)
            ok = response.status_code == 200
        except requests.RequestException:
            ok = False
        return (time.perf_counter() - start) * 1000, ok

    # Warm-up: the first request per worker pays connection setup, and one
    # process-wide warm-up avoids charging that to the measured run.
    for i in range(concurrency):
        one(i)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for ms, ok in pool.map(one, range(n)):
            latencies.append(ms)
            errors += int(not ok)
    elapsed = time.perf_counter() - started

    return {
        "path": path,
        "n": n,
        "errors": errors,
        "rps": n / elapsed,
        "mean": statistics.mean(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "max": max(latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=1000, help="Requests per endpoint.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/load_test.md"))
    args = parser.parse_args()

    try:
        health = requests.get(f"{args.base_url}/health", timeout=5).json()
    except requests.RequestException as exc:
        print(f"error: cannot reach {args.base_url} ({exc})", file=sys.stderr)
        print("start the API first: uvicorn tradetrack.api.main:app --app-dir src", file=sys.stderr)
        return 1

    if health.get("status") != "ok":
        print(f"error: API is '{health.get('status')}' - generate data first", file=sys.stderr)
        return 1

    rows = int(health["trades_loaded"])
    print(f"target {args.base_url} · {rows:,} trades loaded")
    print(f"{args.requests} requests per endpoint at concurrency {args.concurrency}\n")

    results = []
    for path, note in ENDPOINTS:
        result = hammer(args.base_url, path, args.requests, args.concurrency)
        result["note"] = note
        results.append(result)
        print(f"  {path:<36} p50 {result['p50']:6.1f}ms  p95 {result['p95']:6.1f}ms  "
              f"p99 {result['p99']:6.1f}ms  {result['rps']:7.0f} rps  errors {result['errors']}", flush=True)

    lines = [
        "# Load test",
        "",
        f"{args.requests:,} requests per endpoint at concurrency {args.concurrency}, against a ",
        f"snapshot of {rows:,} trades. Regenerate with `python scripts/load_test.py`.",
        "",
        "| Endpoint | p50 | p95 | p99 | max | RPS | Errors | What it costs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['path']}` | {r['p50']:.1f}ms | {r['p95']:.1f}ms | {r['p99']:.1f}ms | "
            f"{r['max']:.1f}ms | {r['rps']:,.0f} | {r['errors']} | {r['note']} |"
        )

    slowest = max(results, key=lambda r: r["p99"])
    fastest = min(results, key=lambda r: r["p99"])
    lines += [
        "",
        "## Reading this honestly",
        "",
        "**Client-observed latency, one machine, loopback, single uvicorn worker.** It",
        "includes client and loopback overhead and excludes real network latency, so it",
        "is a floor on production latency rather than a prediction of it.",
        "",
        f"The spread is the useful part: `{fastest['path']}` sits at {fastest['p99']:.1f}ms p99 while ",
        f"`{slowest['path']}` sits at {slowest['p99']:.1f}ms - roughly "
        f"{slowest['p99'] / max(fastest['p99'], 0.01):.0f}x. That gap is the O(n) mask over the ",
        "snapshot, and it is the measured version of bottleneck #2 in `docs/system_design.md`,",
        "which until now was reasoned about rather than observed.",
        "",
        "One uvicorn worker is one core, so RPS here is a per-worker figure. Scaling out",
        "multiplies it, at the cost of one snapshot copy per worker - which is the memory",
        "constraint `benchmarks/scale_profile.md` measures.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
