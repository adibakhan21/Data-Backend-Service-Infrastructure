"""Pydantic models: the single source of truth for every shape in the system.

The same `Trade` model is used for validation at ingest and for serialisation
at the API boundary, so a field cannot drift between the two.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")


class Side(str, Enum):
    """Direction of an order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Terminal state of an order.

    FILLED and PARTIAL are successes; CANCELLED and REJECTED count as
    order-processing failures for every failure-rate calculation in the
    codebase (see `processing.is_failure`).
    """

    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


FAILURE_STATUSES: frozenset[str] = frozenset({OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value})


class Severity(str, Enum):
    """How strongly an anomaly rule fired, derived from how far past its
    threshold the observed metric sits. Not a probability."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SignalAction(str, Enum):
    """Output of the rule-based signal engine."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Trade(BaseModel):
    """A single order/trade record.

    This is the validation boundary: anything that fails these constraints is
    rejected by the ingest pipeline and counted as a rejected row rather than
    silently written to the database.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    trade_id: str = Field(min_length=1, max_length=64, description="Unique order identifier.")
    timestamp: datetime = Field(description="UTC time the order reached the venue.")
    symbol: str = Field(min_length=1, max_length=16, description="Instrument ticker.")
    side: Side
    quantity: int = Field(gt=0, description="Order size in units; must be positive.")
    price: float = Field(gt=0, description="Limit/execution price; must be positive.")
    order_status: OrderStatus
    execution_latency_ms: float = Field(ge=0, description="Venue round-trip latency.")
    account_id: str = Field(min_length=1, max_length=32)

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, v: str) -> str:
        """Normalise tickers so `aapl` and `AAPL` aggregate together."""
        return v.upper()

    @property
    def notional(self) -> float:
        """Cash value of the order (quantity x price)."""
        return self.quantity * self.price


class SymbolMetrics(BaseModel):
    """Aggregate statistics for one symbol, produced by a single O(n) pass."""

    symbol: str
    trade_count: int
    failed_count: int
    failure_rate: float = Field(description="failed_count / trade_count, in [0, 1].")
    total_quantity: int
    total_notional: float
    vwap: float = Field(description="Volume-weighted average price.")
    avg_latency_ms: float
    p95_latency_ms: float


class GlobalMetrics(BaseModel):
    """Dataset-wide summary shown on the dashboard and `GET /metrics`."""

    total_trades: int
    failed_trades: int
    failure_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_notional: float
    distinct_symbols: int
    distinct_accounts: int
    window_start: datetime | None = None
    window_end: datetime | None = None


class Anomaly(BaseModel):
    """One explainable anomaly finding.

    Every field exists so a human can audit the decision: which rule fired,
    on what entity, which metric, and how far past which threshold.
    """

    anomaly_type: str = Field(description="Rule identifier, e.g. HIGH_LATENCY.")
    severity: Severity
    scope: str = Field(description="Entity kind the rule applies to: trade | symbol | account.")
    entity_id: str = Field(description="trade_id, symbol or account_id depending on scope.")
    symbol: str | None = None
    reason: str = Field(description="Plain-English explanation of why this fired.")
    metric_name: str
    metric_value: float
    threshold: float


class Signal(BaseModel):
    """A rule-based trading signal.

    Deliberately not a prediction: no backtest, no profitability claim. These
    rules describe what recent synthetic order flow did, nothing more.
    """

    symbol: str
    action: SignalAction
    confidence: float = Field(ge=0, le=1, description="Rule agreement score, not a probability of profit.")
    reason: str
    momentum_pct: float = Field(description="Recent VWAP vs prior VWAP, in percent.")
    volume_ratio: float = Field(description="Recent volume vs prior volume.")
    sample_size: int


class Page(BaseModel, Generic[T]):
    """Envelope for every list endpoint, so clients page uniformly."""

    items: list[T]
    total: int = Field(description="Total rows matching the filter, ignoring pagination.")
    limit: int
    offset: int
    has_more: bool
