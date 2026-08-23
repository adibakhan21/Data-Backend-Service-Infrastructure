# Load test

800 requests per endpoint at concurrency 16, against a 
snapshot of 200,000 trades. Regenerate with `python scripts/load_test.py`.

| Endpoint | p50 | p95 | p99 | max | RPS | Errors | What it costs |
|---|---|---|---|---|---|---|---|
| `/health` | 4.0ms | 6.7ms | 8.0ms | 10.1ms | 3,813 | 0 | liveness, no data access |
| `/metrics` | 4.0ms | 6.7ms | 7.7ms | 10.0ms | 3,745 | 0 | precomputed, O(1) read |
| `/trades?limit=100` | 18.6ms | 24.5ms | 35.0ms | 40.4ms | 836 | 0 | O(n) mask + serialise 100 rows |
| `/trades?symbol=NVDA&limit=50` | 25.3ms | 32.3ms | 34.9ms | 40.6ms | 620 | 0 | O(n) filtered mask |
| `/trades/T0000000042` | 4.2ms | 6.5ms | 7.9ms | 9.9ms | 3,570 | 0 | O(1) hash lookup |
| `/metrics/top?sort_by=notional&k=5` | 4.2ms | 6.6ms | 8.2ms | 9.2ms | 3,639 | 0 | heap top-K |
| `/anomalies?limit=50` | 4.7ms | 6.9ms | 8.1ms | 12.6ms | 3,276 | 0 | filter + slice |

## Reading this honestly

**Client-observed latency, one machine, loopback, single uvicorn worker.** It
includes client and loopback overhead and excludes real network latency, so it
is a floor on production latency rather than a prediction of it.

The spread is the useful part: `/metrics` sits at 7.7ms p99 while 
`/trades?limit=100` sits at 35.0ms - roughly 5x. That gap is the O(n) mask over the 
snapshot, and it is the measured version of bottleneck #2 in `docs/system_design.md`,
which until now was reasoned about rather than observed.

One uvicorn worker is one core, so RPS here is a per-worker figure. Scaling out
multiplies it, at the cost of one snapshot copy per worker - which is the memory
constraint `benchmarks/scale_profile.md` measures.
