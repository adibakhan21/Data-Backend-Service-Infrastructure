"""Algorithm correctness, including the edge cases that break naive versions."""

from __future__ import annotations

import numpy as np
import pytest

from tradetrack.algorithms import (
    build_index,
    group_reduce,
    index_lookup,
    linear_lookup,
    median_absolute_deviation,
    naive_top_k,
    naive_window_failure_rate,
    percentile,
    sliding_window_failure_rate,
    top_k,
)


class TestIndexLookup:
    def test_finds_existing_key(self):
        records = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
        index = build_index(records, key=lambda r: r["id"])
        assert index_lookup(index, "b") == {"id": "b", "v": 2}

    def test_missing_key_returns_none(self):
        assert index_lookup(build_index([], key=lambda r: r), "nope") is None

    def test_agrees_with_linear_scan(self):
        records = [{"id": f"k{i}"} for i in range(500)]
        index = build_index(records, key=lambda r: r["id"])
        for target in ("k0", "k250", "k499", "missing"):
            assert index_lookup(index, target) == linear_lookup(records, lambda r: r["id"], target)

    def test_duplicate_keys_keep_the_last(self):
        """Documented behaviour, asserted so it cannot change silently."""
        records = [{"id": "x", "v": 1}, {"id": "x", "v": 2}]
        assert build_index(records, key=lambda r: r["id"])["x"]["v"] == 2


class TestGroupReduce:
    def test_counts_by_group(self):
        items = ["a", "b", "a", "c", "a"]
        counts = group_reduce(items, key=lambda s: s, initial=lambda: 0, reducer=lambda acc, _: acc + 1)
        assert counts == {"a": 3, "b": 1, "c": 1}

    def test_empty_input(self):
        assert group_reduce([], key=lambda s: s, initial=lambda: 0, reducer=lambda a, _: a) == {}

    def test_falsy_accumulator_is_not_reinitialised(self):
        """Regression guard: `if not current` instead of `is None` would reset
        every group whose running total happened to be 0."""
        items = [0, 0, 5]
        total = group_reduce(items, key=lambda _: "k", initial=lambda: 0, reducer=lambda acc, v: acc + v)
        assert total == {"k": 5}


class TestSlidingWindow:
    def test_matches_naive_implementation(self):
        rng = np.random.default_rng(0)
        flags = (rng.random(2_000) < 0.2).tolist()
        for window in (1, 7, 100, 500):
            fast = sliding_window_failure_rate(flags, window)
            slow = naive_window_failure_rate(flags, window)
            assert fast == pytest.approx(slow, abs=1e-12)

    def test_window_larger_than_input(self):
        assert sliding_window_failure_rate([True, False], 100) == [1.0, 0.5]

    def test_all_failures_and_none(self):
        assert sliding_window_failure_rate([True] * 5, 3) == [1.0] * 5
        assert sliding_window_failure_rate([False] * 5, 3) == [0.0] * 5

    def test_empty_input(self):
        assert sliding_window_failure_rate([], 10) == []

    @pytest.mark.parametrize("window", [0, -1])
    def test_invalid_window_raises(self, window):
        with pytest.raises(ValueError, match="positive"):
            sliding_window_failure_rate([True], window)


class TestTopK:
    def test_matches_full_sort(self):
        rng = np.random.default_rng(1)
        items = [{"v": float(x)} for x in rng.normal(size=1_000)]
        key = lambda d: d["v"]  # noqa: E731
        for k in (1, 5, 50):
            assert [d["v"] for d in top_k(items, key, k)] == [d["v"] for d in naive_top_k(items, key, k)]

    def test_k_larger_than_input(self):
        assert len(top_k([1, 2, 3], lambda x: x, 10)) == 3

    @pytest.mark.parametrize("k", [0, -3])
    def test_non_positive_k_returns_empty(self, k):
        assert top_k([1, 2, 3], lambda x: x, k) == []

    def test_empty_input(self):
        assert top_k([], lambda x: x, 5) == []

    def test_handles_unorderable_items(self):
        """Items are never compared directly - only their keys - so types with
        no `__lt__` must still work. Without the tie-break counter this raises
        TypeError whenever two keys are equal."""
        items = [{"v": 1}, {"v": 1}, {"v": 1}]
        assert len(top_k(items, lambda d: d["v"], 2)) == 2


class TestPercentile:
    def test_matches_numpy(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            values = rng.lognormal(3, 0.6, size=int(rng.integers(2, 1_000))).tolist()
            for q in (0, 25, 50, 95, 99, 100):
                assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)), abs=1e-9)

    def test_single_element(self):
        assert percentile([42.0], 99) == 42.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            percentile([], 50)

    @pytest.mark.parametrize("pct", [-1, 101])
    def test_out_of_range_raises(self, pct):
        with pytest.raises(ValueError, match="between 0 and 100"):
            percentile([1.0, 2.0], pct)


class TestMedianAbsoluteDeviation:
    def test_known_values(self):
        assert median_absolute_deviation([1, 2, 3, 4, 5]) == (3.0, 1.0)

    def test_resists_outliers(self):
        """The property the quantity rule depends on: one extreme value moves
        the standard deviation enormously and the MAD barely at all."""
        clean = [10.0] * 50 + [12.0] * 50
        polluted = clean + [10_000.0]
        assert median_absolute_deviation(polluted)[1] == pytest.approx(
            median_absolute_deviation(clean)[1], abs=1.0
        )
        assert np.std(polluted) > 20 * np.std(clean)

    def test_zero_spread(self):
        assert median_absolute_deviation([7.0] * 10) == (7.0, 0.0)
