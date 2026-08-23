"""Signal rules.

These tests assert the rules behave as documented. They deliberately do not
assert anything about predictive accuracy - there is none to measure, and a
test claiming otherwise would be the exact fabrication this project avoids.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradetrack.config import Settings
from tradetrack.models import SignalAction
from tradetrack.signals import generate_signals


def _series(prices: list[float], quantities: list[int] | None = None) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame({
        "trade_id": [f"T{i}" for i in range(n)],
        "timestamp": pd.date_range("2025-01-06", periods=n, freq="s", tz="UTC"),
        "symbol": ["AAPL"] * n,
        "side": ["BUY"] * n,
        "quantity": quantities or [100] * n,
        "price": prices,
        "order_status": ["FILLED"] * n,
        "execution_latency_ms": [10.0] * n,
        "account_id": ["ACC1"] * n,
    })


CONFIG = Settings(signal_window=10)


class TestSignalRules:
    def test_rising_vwap_on_steady_volume_gives_buy(self):
        signals = generate_signals(_series([100.0] * 10 + [110.0] * 10), CONFIG)
        assert signals[0].action is SignalAction.BUY
        assert signals[0].momentum_pct == pytest.approx(10.0)

    def test_falling_vwap_gives_sell(self):
        signals = generate_signals(_series([110.0] * 10 + [100.0] * 10), CONFIG)
        assert signals[0].action is SignalAction.SELL
        assert signals[0].momentum_pct < 0

    def test_flat_price_gives_hold(self):
        signals = generate_signals(_series([100.0] * 20), CONFIG)
        assert signals[0].action is SignalAction.HOLD
        assert signals[0].confidence == 0.0

    def test_price_move_on_collapsing_volume_gives_hold(self):
        """The volume guard: a move on a thin tape is the least trustworthy
        kind, so the rules refuse to signal on it."""
        signals = generate_signals(_series([100.0] * 10 + [110.0] * 10, [1000] * 10 + [10] * 10), CONFIG)
        assert signals[0].action is SignalAction.HOLD

    def test_vwap_weights_by_size_not_by_count(self):
        """One 10,000-lot at 200 must move the VWAP far more than nine 1-lots
        at 100. A mean price would treat them as near-equals."""
        signals = generate_signals(
            _series([100.0] * 10 + [100.0] * 9 + [200.0], [100] * 10 + [1] * 9 + [10_000]), CONFIG
        )
        assert signals[0].momentum_pct > 50


class TestSignalGuards:
    def test_symbols_below_two_windows_are_skipped(self):
        """19 trades with window=10 is not enough for a prior and a recent
        period, so no signal is emitted at all - not a low-confidence guess."""
        assert generate_signals(_series([100.0] * 19), CONFIG) == []

    def test_confidence_is_bounded(self, frame):
        for signal in generate_signals(frame):
            assert 0.0 <= signal.confidence <= 1.0

    def test_sorted_by_confidence(self, frame):
        confidences = [s.confidence for s in generate_signals(frame)]
        assert confidences == sorted(confidences, reverse=True)

    def test_every_signal_explains_itself(self, frame):
        for signal in generate_signals(frame):
            assert signal.reason and signal.sample_size > 0

    def test_one_signal_per_symbol(self, frame):
        signals = generate_signals(frame)
        assert len({s.symbol for s in signals}) == len(signals)

    def test_empty_input(self, empty_frame):
        assert generate_signals(empty_frame) == []
