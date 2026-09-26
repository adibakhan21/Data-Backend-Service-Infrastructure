# Benchmark results

Machine: `arm64 / Darwin 24.6.0 / Python 3.13.7`  
Method: fastest of 3 runs per case (minimum, not mean - see the
script docstring for why).

**These runtimes are measurements from one machine on one day. The complexity
column is the part that transfers.** Regenerate with `python scripts/run_benchmarks.py`.

| Benchmark | n | Baseline | Optimised | Baseline | Optimised | Speedup | Complexity change |
|---|---|---|---|---|---|---|---|
| Trade lookup by id (worst case: last row) | 200,000 | linear scan | hash index | 6.568 ms | 22.1 ns | 296,594.4x | O(n) -> O(1) average. Index build is a one-time O(n). |
| Top-10 by quantity | 200,000 | full sort | size-k heap | 18.757 ms | 14.219 ms | 1.3x | O(n log n) -> O(n log k) time; O(n) -> O(k) space at k=10. |
| Rolling failure rate (window=500) | 50,000 | recompute window | running sum | 75.558 ms | 3.450 ms | 21.9x | O(n*w) -> O(n) time at w=500; O(n) -> O(w) space. |
| Synthetic data generation | 100,000 | row-by-row Python | vectorised numpy | 809.066 ms | 49.112 ms | 16.5x | Both O(n). Constant-factor only: the per-row work moves from the Python interpreter into C. |
| SQLite bulk insert | 20,000 | commit per row | executemany, one txn | 6.242 s | 124.925 ms | 50.0x | Both O(n log n) for index maintenance. The win is fsync count: n commits -> 1. |

## Reading these honestly

* **Lookup** and **rolling failure rate** are genuine complexity wins: the speedup
  grows with n, so it would be larger on a bigger dataset and smaller on a smaller one.
  The lookup ratio is large because it is O(n) vs O(1) at n=200,000 - quoting it as a
  bare multiplier is meaningless without that n. The useful statement is the absolute
  one: a full scan of 200,000 records costs milliseconds, a dict hit costs nanoseconds.
* **Top-K** at 12 symbols would show nothing; it is benchmarked over all trades to make
  the effect visible. In the API the input is the symbol count, where the sort would be fine.
  The heap is there for the case where the group count is large, not for today's 12.
* **Generation** and **SQLite insert** are constant-factor wins, not complexity wins.
  Real, useful, and they would not change the shape of the scaling curve.
