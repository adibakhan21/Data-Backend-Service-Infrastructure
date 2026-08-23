"""Signal listing.

Reminder, repeated here because it matters at the API boundary: these are
descriptive rules over synthetic flow with no backtest behind them. See the
module docstring in `tradetrack/signals.py`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from ...models import Page, Signal, SignalAction
from ..deps import PageDep, StoreDep

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("", response_model=Page[Signal], summary="List rule-based signals")
def list_signals(
    store: StoreDep,
    page: PageDep,
    action: Annotated[SignalAction | None, Query(description="BUY, SELL or HOLD.")] = None,
    min_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
    symbol: Annotated[str | None, Query(description="Restrict to one ticker.")] = None,
) -> Page[Signal]:
    """Filter and page over the precomputed, confidence-sorted signal list."""
    items = store.signals
    if action:
        items = [s for s in items if s.action == action]
    if min_confidence is not None:
        items = [s for s in items if s.confidence >= min_confidence]
    if symbol:
        wanted = symbol.upper()
        items = [s for s in items if s.symbol == wanted]

    total = len(items)
    return Page[Signal](
        items=items[page.offset : page.offset + page.limit],
        total=total,
        limit=page.limit,
        offset=page.offset,
        has_more=page.offset + page.limit < total,
    )
