"""Aggregate metrics: dataset-wide, per symbol, and top-K."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query

from ...algorithms import top_k
from ...models import GlobalMetrics, SymbolMetrics
from ..deps import StoreDep
from ..errors import ErrorResponse, SymbolNotFoundError

router = APIRouter(prefix="/metrics", tags=["metrics"])

# Which SymbolMetrics field each `sort_by` value maps to. Keeping this as an
# explicit allow-list rather than `getattr(metrics, user_string)` is the point:
# a user-supplied attribute name is an injection primitive.
SORT_FIELDS: dict[str, str] = {
    "trade_count": "trade_count",
    "notional": "total_notional",
    "quantity": "total_quantity",
    "failure_rate": "failure_rate",
    "latency": "p95_latency_ms",
}


@router.get("", response_model=GlobalMetrics, summary="Dataset-wide summary")
def get_global_metrics(store: StoreDep) -> GlobalMetrics:
    """Precomputed at snapshot build time, so this is an O(1) attribute read."""
    assert store.metrics is not None  # guaranteed by AnalyticsStore.build
    return store.metrics


@router.get("/top", response_model=list[SymbolMetrics], summary="Top-K symbols by a metric")
def get_top_symbols(
    store: StoreDep,
    sort_by: Annotated[str, Query(description=f"One of: {', '.join(SORT_FIELDS)}")] = "notional",
    k: Annotated[int, Query(ge=1, le=100, description="How many symbols to return.")] = 5,
) -> list[SymbolMetrics]:
    """Heap-based top-K: O(g log k) over g symbols, not O(g log g).

    At g=12 symbols the difference is unmeasurable and a sort would be fine.
    The heap is here because g is the thing that grows - a real venue has tens
    of thousands of instruments, and the same endpoint over accounts would run
    over hundreds of thousands. `scripts/run_benchmarks.py` measures the gap at
    sizes where it is visible.
    """
    if sort_by not in SORT_FIELDS:
        raise SymbolNotFoundError(sort_by)
    field = SORT_FIELDS[sort_by]
    return top_k(store.symbol_metrics.values(), key=lambda m: getattr(m, field), k=k)


@router.get(
    "/symbol/{symbol}",
    response_model=SymbolMetrics,
    summary="Metrics for one symbol",
    responses={404: {"model": ErrorResponse, "description": "Unknown symbol"}},
)
def get_symbol_metrics(
    store: StoreDep,
    symbol: Annotated[str, Path(min_length=1, max_length=16)],
) -> SymbolMetrics:
    """O(1) average dict lookup into the precomputed per-symbol aggregates."""
    metrics = store.symbol_metrics.get(symbol.upper())
    if metrics is None:
        raise SymbolNotFoundError(symbol)
    return metrics
