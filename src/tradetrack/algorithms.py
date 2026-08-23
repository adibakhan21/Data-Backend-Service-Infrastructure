"""Core algorithmic building blocks, with their complexity stated honestly.

Every function here is pure: it takes plain sequences/mappings and returns
plain data. That makes them trivial to unit-test and to benchmark against a
naive counterpart, which is exactly what `scripts/run_benchmarks.py` does.

Naive counterparts are kept in this file on purpose. They are not dead code:
they are the control arm of the benchmark, and deleting them would make the
measured speedups unreproducible.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


# --------------------------------------------------------------------------
# 1. Hash-map lookup: O(n) to build, O(1) average per lookup
# --------------------------------------------------------------------------


def build_index(records: Iterable[T], key: Callable[[T], K]) -> dict[K, T]:
    """Build a hash index from a key function.

    Complexity:
        Time  O(n) - one pass, each insert O(1) average.
        Space O(n) - one dict entry per record.

    On collisions the *last* record wins; callers that need first-wins should
    deduplicate upstream.
    """
    return {key(record): record for record in records}


def index_lookup(index: Mapping[K, T], key: K) -> T | None:
    """Look up one record by key.

    Complexity:
        Time O(1) *average*. This is not a worst-case guarantee: CPython dicts
        degrade to O(n) if every key lands in the same bucket. With random
        string trade IDs that does not happen in practice, but the honest
        statement is "O(1) amortised average", not "O(1)".
    """
    return index.get(key)


def linear_lookup(records: Sequence[T], key: Callable[[T], K], target: K) -> T | None:
    """Naive scan-until-found lookup - the control arm for the benchmark.

    Complexity:
        Time O(n) - touches every record until a match. Average n/2 for a hit,
        always n for a miss.
    """
    for record in records:
        if key(record) == target:
            return record
    return None


# --------------------------------------------------------------------------
# 2. Hash-map aggregation: O(n)
# --------------------------------------------------------------------------


def group_reduce(
    records: Iterable[T],
    key: Callable[[T], K],
    initial: Callable[[], Any],
    reducer: Callable[[Any, T], Any],
) -> dict[K, Any]:
    """Single-pass grouped reduction (the map-side of a group-by).

    Complexity:
        Time  O(n) - one pass; each step is a dict get (O(1) average) plus the
              cost of `reducer`, which must itself be O(1) for this bound to
              hold.
        Space O(g) where g is the number of distinct groups - crucially *not*
              O(n). This is why aggregation scales to datasets far larger than
              memory would allow if we kept the rows themselves.

    Note this is strictly better than sorting-then-grouping, which is
    O(n log n). We never need the groups in sorted order, so we never pay for
    the sort.
    """
    accumulator: dict[K, Any] = {}
    for record in records:
        group = key(record)
        current = accumulator.get(group)
        if current is None:
            current = initial()
            accumulator[group] = current
        accumulator[group] = reducer(current, record)
    return accumulator


# --------------------------------------------------------------------------
# 3. Sliding-window failure rate: O(n) total, O(w) space
# --------------------------------------------------------------------------


def sliding_window_failure_rate(
    flags: Sequence[bool], window: int
) -> list[float]:
    """Rolling failure rate over the last `window` observations.

    `flags[i]` is True when observation i was a failure. Returns a list the
    same length as `flags`; entry i is the failure rate over the window ending
    at i (shorter than `window` at the start, by design, so early entries are
    still usable rather than NaN).

    Complexity:
        Time  O(n) - each element enters the deque once and leaves once, and
              the running sum is updated in O(1). The naive version recomputes
              sum(window) at every step and is O(n * w).
        Space O(w) - the deque holds at most `window` booleans.

    Raises:
        ValueError: if `window` is not positive.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")

    buffer: deque[bool] = deque(maxlen=window)
    failures = 0
    rates: list[float] = []

    for flag in flags:
        if len(buffer) == window:
            # deque(maxlen=) will evict the oldest element on append, so we
            # must discount it from the running sum *before* appending.
            if buffer[0]:
                failures -= 1
        buffer.append(flag)
        if flag:
            failures += 1
        rates.append(failures / len(buffer))

    return rates


def naive_window_failure_rate(flags: Sequence[bool], window: int) -> list[float]:
    """Recompute-the-whole-window version - the control arm for the benchmark.

    Complexity:
        Time O(n * w) - re-sums up to `window` elements at every position.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")
    rates: list[float] = []
    for i in range(len(flags)):
        chunk = flags[max(0, i - window + 1) : i + 1]
        rates.append(sum(chunk) / len(chunk))
    return rates


# --------------------------------------------------------------------------
# 4. Top-K with a heap: O(n log k)
# --------------------------------------------------------------------------


def top_k(items: Iterable[T], key: Callable[[T], float], k: int) -> list[T]:
    """Return the k largest items by `key`, descending.

    Complexity:
        Time  O(n log k) - we keep a min-heap of size k. Each of the n items
              costs at most one O(log k) push plus one O(log k) pop. The final
              sort of k elements is O(k log k), which is dominated by the scan
              whenever k << n.
        Space O(k) - only k items are ever held, never all n.

    Compare `naive_top_k`, which is O(n log n) time *and* O(n) space because it
    sorts everything to keep k. When k=10 and n=10M, log2(k) is ~3.3 while
    log2(n) is ~23 - roughly a 7x reduction in comparison work, and the space
    difference is the bigger practical win.

    Ties are broken by insertion order via a monotonically increasing counter,
    which also keeps the heap from ever comparing the items themselves (items
    may not be orderable).
    """
    if k <= 0:
        return []

    heap: list[tuple[float, int, T]] = []
    for counter, item in enumerate(items):
        entry = (key(item), counter, item)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            # Cheaper than push-then-pop: one sift-down instead of two sifts.
            heapq.heapreplace(heap, entry)

    heap.sort(key=lambda entry: entry[0], reverse=True)
    return [entry[2] for entry in heap]


def naive_top_k(items: Iterable[T], key: Callable[[T], float], k: int) -> list[T]:
    """Sort-everything version - the control arm for the benchmark.

    Complexity:
        Time  O(n log n)
        Space O(n) - materialises and sorts the full list.
    """
    if k <= 0:
        return []
    return sorted(items, key=key, reverse=True)[:k]


# --------------------------------------------------------------------------
# 5. Percentiles: O(n log n) via sort
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile, matching numpy's default method.

    Complexity:
        Time  O(n log n) - dominated by the sort.
        Space O(n) - sorts a copy, leaving the caller's sequence untouched.

    A selection algorithm (quickselect) would give O(n) average for a single
    percentile, but we routinely need p95 and p99 from the same array and the
    sorted copy is reused, so the sort is the cheaper choice here.

    Raises:
        ValueError: if `values` is empty or `pct` is outside [0, 100].
    """
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    lo, hi = ordered[low], ordered[high]
    # Written as `lo + (hi - lo) * w` rather than the algebraically equal
    # `lo*(1-w) + hi*w` so the result is bit-identical to numpy's default
    # 'linear' method. The two forms round differently in the last bit, and
    # that was enough to make the pure-Python and pandas aggregation paths
    # disagree after rounding to 3 decimals.
    return float(lo + (hi - lo) * weight)


def median_absolute_deviation(values: Sequence[float]) -> tuple[float, float]:
    """Return (median, MAD) - a robust alternative to (mean, stdev).

    MAD is used by the unusual-quantity rule because order sizes are heavily
    right-skewed: a handful of block trades drag the mean and standard
    deviation up so far that the rule stops firing at all. The median and MAD
    barely move, so the rule keeps working on skewed data.

    Complexity:
        Time  O(n log n) - two sorts (one for each median).
        Space O(n)
    """
    if not values:
        raise ValueError("MAD of an empty sequence is undefined")
    med = percentile(values, 50)
    deviations = [abs(v - med) for v in values]
    return med, percentile(deviations, 50)
