"""Vectorised synthetic trade generator.

The whole dataset is produced with numpy array operations rather than a Python
loop over record objects. That is the difference between "generate 5M rows" and
"generate 5M rows before lunch": every column is one C-level array op, so the
per-row Python interpreter overhead disappears entirely.

`generate_rows_naive` keeps the row-by-row implementation for benchmarking. It
is the control arm, not dead code.

Anomalies are *injected deliberately* so the detector has something true to
find. `generate` returns the ground-truth injection plan alongside the data,
which is what lets `tests/test_anomalies.py` assert real recall instead of
just "it returned something".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# A small universe keeps per-symbol groups large enough for the statistics to
# mean anything at the default 100k rows.
SYMBOLS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
    "JPM", "XOM", "WMT", "INFY", "TCS",
)

# Rough starting prices, only so notionals land in a believable range.
BASE_PRICES: dict[str, float] = {
    "AAPL": 195.0, "MSFT": 415.0, "NVDA": 880.0, "TSLA": 175.0,
    "AMZN": 180.0, "GOOGL": 165.0, "META": 490.0, "JPM": 195.0,
    "XOM": 115.0, "WMT": 60.0, "INFY": 18.0, "TCS": 45.0,
}

STATUSES: tuple[str, ...] = ("FILLED", "PARTIAL", "CANCELLED", "REJECTED")
BASE_STATUS_WEIGHTS: tuple[float, ...] = (0.86, 0.06, 0.05, 0.03)


@dataclass
class InjectionPlan:
    """Ground truth about what was deliberately made anomalous.

    Attributes:
        latency_trade_ids: Trades given a multi-second latency spike.
        failing_accounts: Accounts whose failure probability was raised.
        oversized_trade_ids: Trades given an order size far above the symbol's
            normal distribution.
    """

    latency_trade_ids: list[str] = field(default_factory=list)
    failing_accounts: list[str] = field(default_factory=list)
    oversized_trade_ids: list[str] = field(default_factory=list)


@dataclass
class GeneratorConfig:
    """Knobs for the generator.

    Attributes:
        n_rows: Number of trades to produce.
        n_accounts: Size of the account pool.
        seed: RNG seed; the same seed always yields byte-identical data.
        start: Timestamp of the first trade (UTC).
        mean_gap_ms: Average inter-arrival time between consecutive trades.
        latency_spike_rate: Fraction of trades given a latency spike.
        oversized_rate: Fraction of trades given an outsized quantity.
        n_failing_accounts: How many accounts get an elevated failure rate.
        failing_account_multiplier: How much their failure odds are multiplied.
    """

    n_rows: int = 100_000
    n_accounts: int = 200
    seed: int = 42
    start: datetime = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
    mean_gap_ms: float = 20.0
    latency_spike_rate: float = 0.004
    oversized_rate: float = 0.002
    n_failing_accounts: int = 4
    failing_account_multiplier: float = 6.0


def generate(config: GeneratorConfig | None = None) -> tuple[pd.DataFrame, InjectionPlan]:
    """Generate a synthetic trade dataset.

    Returns:
        (frame, plan) where `frame` has one row per trade with the schema in
        `models.Trade`, and `plan` records which rows were deliberately made
        anomalous.

    Complexity:
        Time  O(n) with a very small constant - every column is a single
              vectorised numpy call. The one O(n log n) step is the timestamp
              cumulative sum's implicit ordering, which is actually O(n).
        Space O(n) - the frame is materialised in full. For datasets beyond
              memory, call this in chunks (see `scripts/generate_data.py`,
              which streams chunks straight into SQLite).
    """
    cfg = config or GeneratorConfig()
    if cfg.n_rows <= 0:
        raise ValueError("n_rows must be positive")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_rows

    trade_ids = np.array([f"T{i:010d}" for i in range(n)], dtype=object)

    # Inter-arrival times ~ Exponential, so trade timestamps form a Poisson
    # process - bursty, like a real tape, rather than evenly spaced.
    gaps_ms = rng.exponential(cfg.mean_gap_ms, size=n)
    offsets_ms = np.cumsum(gaps_ms)
    timestamps = pd.to_datetime(cfg.start) + pd.to_timedelta(offsets_ms, unit="ms")

    # Zipf-ish popularity: a few symbols carry most of the flow.
    symbol_weights = 1.0 / np.arange(1, len(SYMBOLS) + 1) ** 0.7
    symbol_weights /= symbol_weights.sum()
    symbol_idx = rng.choice(len(SYMBOLS), size=n, p=symbol_weights)
    symbols = np.array(SYMBOLS, dtype=object)[symbol_idx]

    sides = np.where(rng.random(n) < 0.5, "BUY", "SELL").astype(object)

    accounts = np.array([f"ACC{i:05d}" for i in range(cfg.n_accounts)], dtype=object)
    account_idx = rng.integers(0, cfg.n_accounts, size=n)
    account_ids = accounts[account_idx]

    # Prices: per-symbol base with a random walk of small relative moves.
    base = np.array([BASE_PRICES[s] for s in SYMBOLS])[symbol_idx]
    drift = np.cumsum(rng.normal(0.0, 0.0004, size=n))
    prices = np.round(base * (1.0 + drift + rng.normal(0.0, 0.0015, size=n)), 2)
    prices = np.maximum(prices, 0.01)  # a price must stay strictly positive

    # Order sizes are heavily right-skewed, hence lognormal not normal.
    quantities = np.maximum(1, rng.lognormal(mean=4.2, sigma=1.0, size=n).astype(np.int64))

    # ---- deliberate anomaly injection -----------------------------------
    plan = InjectionPlan()

    oversized_mask = rng.random(n) < cfg.oversized_rate
    quantities[oversized_mask] *= rng.integers(60, 200, size=int(oversized_mask.sum()))
    plan.oversized_trade_ids = trade_ids[oversized_mask].tolist()

    # Baseline latency is lognormal (a long right tail is normal for venues).
    latencies = rng.lognormal(mean=3.1, sigma=0.55, size=n)
    spike_mask = rng.random(n) < cfg.latency_spike_rate
    latencies[spike_mask] *= rng.uniform(25, 120, size=int(spike_mask.sum()))
    plan.latency_trade_ids = trade_ids[spike_mask].tolist()
    latencies = np.round(latencies, 2)

    # Status: baseline weights, then a few accounts get much worse odds.
    failing_accounts = accounts[rng.choice(cfg.n_accounts, size=cfg.n_failing_accounts, replace=False)]
    plan.failing_accounts = failing_accounts.tolist()
    failing_set = set(plan.failing_accounts)
    is_failing_account = np.array([acc in failing_set for acc in account_ids])

    statuses = np.array(STATUSES, dtype=object)[
        rng.choice(len(STATUSES), size=n, p=BASE_STATUS_WEIGHTS)
    ]
    # Re-roll only the rows belonging to bad accounts, with skewed weights.
    bad_weights = np.array(BASE_STATUS_WEIGHTS, dtype=float).copy()
    bad_weights[2:] *= cfg.failing_account_multiplier
    bad_weights /= bad_weights.sum()
    n_bad = int(is_failing_account.sum())
    if n_bad:
        statuses[is_failing_account] = np.array(STATUSES, dtype=object)[
            rng.choice(len(STATUSES), size=n_bad, p=bad_weights)
        ]

    frame = pd.DataFrame(
        {
            "trade_id": trade_ids,
            "timestamp": timestamps,
            "symbol": symbols,
            "side": sides,
            "quantity": quantities,
            "price": prices,
            "order_status": statuses,
            "execution_latency_ms": latencies,
            "account_id": account_ids,
        }
    )
    return frame, plan


def generate_rows_naive(config: GeneratorConfig | None = None) -> list[dict]:
    """Row-by-row generator - the control arm for the generation benchmark.

    Produces statistically similar (not identical) data using per-row Python
    calls into the RNG. Complexity is the same O(n); the constant factor is
    what differs, and that constant is the entire point of the benchmark.
    """
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict] = []
    clock = cfg.start
    for i in range(cfg.n_rows):
        clock = clock + timedelta(milliseconds=float(rng.exponential(cfg.mean_gap_ms)))
        symbol = SYMBOLS[int(rng.integers(0, len(SYMBOLS)))]
        rows.append(
            {
                "trade_id": f"T{i:010d}",
                "timestamp": clock,
                "symbol": symbol,
                "side": "BUY" if rng.random() < 0.5 else "SELL",
                "quantity": max(1, int(rng.lognormal(4.2, 1.0))),
                "price": round(float(BASE_PRICES[symbol] * (1 + rng.normal(0, 0.0015))), 2),
                "order_status": str(rng.choice(STATUSES, p=BASE_STATUS_WEIGHTS)),
                "execution_latency_ms": round(float(rng.lognormal(3.1, 0.55)), 2),
                "account_id": f"ACC{int(rng.integers(0, cfg.n_accounts)):05d}",
            }
        )
    return rows
