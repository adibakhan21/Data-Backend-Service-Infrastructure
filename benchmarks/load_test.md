# Load test

1,000 requests per endpoint at concurrency 16, against a 
snapshot of 200,000 trades. Regenerate with `python scripts/load_test.py`.

| Endpoint | p50 | p95 | p99 | max | RPS | Errors | What it costs |
|---|---|---|---|---|---|---|---|
| `/health` | 4.0ms | 6.6ms | 8.2ms | 9.5ms | 3,783 | 0 | liveness, no data access |
| `/metrics` | 4.1ms | 6.6ms | 7.6ms | 11.6ms | 3,724 | 0 | precomputed, O(1) read |
| `/trades?limit=100` | 18.4ms | 23.3ms | 31.5ms | 34.6ms | 857 | 0 | O(n) mask + serialise 100 rows |
| `/trades?symbol=NVDA&limit=50` | 25.7ms | 31.9ms | 34.6ms | 36.5ms | 616 | 0 | O(n) filtered mask |
| `/trades/T0000000042` | 4.1ms | 6.9ms | 19.5ms | 24.3ms | 3,467 | 0 | O(1) hash lookup |
| `/metrics/top?sort_by=notional&k=5` | 4.1ms | 6.8ms | 7.8ms | 9.8ms | 3,675 | 0 | heap top-K |
| `/anomalies?limit=50` | 4.4ms | 6.3ms | 7.4ms | 10.0ms | 3,517 | 0 | filter + slice |

## Reading this honestly

**Client-observed latency, one machine, loopback, single uvicorn worker.** It
includes client and loopback overhead and excludes real network latency, so it
is a floor on production latency rather than a prediction of it.

The spread is the useful part: `/anomalies?limit=50` sits at 7.4ms p99 while 
`/trades?symbol=NVDA&limit=50` sits at 34.6ms - roughly 5x. That gap is the O(n) mask over the 
snapshot, and it is the measured version of bottleneck #2 in `docs/system_design.md`,
which until now was reasoned about rather than observed.

One uvicorn worker is one core, so RPS here is a per-worker figure. Scaling out
multiplies it, at the cost of one snapshot copy per worker - which is the memory
constraint `benchmarks/scale_profile.md` measures.
