"""Anomaly listing."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from ...models import Anomaly, Page, Severity
from ..deps import PageDep, StoreDep

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("", response_model=Page[Anomaly], summary="List detected anomalies")
def list_anomalies(
    store: StoreDep,
    page: PageDep,
    anomaly_type: Annotated[
        str | None,
        Query(description="HIGH_LATENCY | HIGH_FAILURE_RATE | UNUSUAL_QUANTITY"),
    ] = None,
    severity: Annotated[Severity | None, Query(description="Exact severity band.")] = None,
    symbol: Annotated[str | None, Query(description="Restrict to one ticker.")] = None,
) -> Page[Anomaly]:
    """Already severity-sorted by the store, so this is filter + slice only.

    Complexity: O(a) over the anomaly list, which is far smaller than the trade
    count - the detection itself happened once at snapshot build.
    """
    items = store.anomalies
    if anomaly_type:
        wanted = anomaly_type.upper()
        items = [a for a in items if a.anomaly_type == wanted]
    if severity:
        items = [a for a in items if a.severity == severity]
    if symbol:
        wanted_symbol = symbol.upper()
        items = [a for a in items if a.symbol == wanted_symbol]

    total = len(items)
    return Page[Anomaly](
        items=items[page.offset : page.offset + page.limit],
        total=total,
        limit=page.limit,
        offset=page.offset,
        has_more=page.offset + page.limit < total,
    )
