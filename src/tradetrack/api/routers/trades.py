"""Trade listing and single-trade lookup."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Path, Query

from ...models import OrderStatus, Page, Side, Trade
from ..deps import PageDep, StoreDep
from ..errors import ErrorResponse, TradeNotFoundError

router = APIRouter(prefix="/trades", tags=["trades"])


@router.get("", response_model=Page[Trade], summary="List trades with filters and pagination")
def list_trades(
    store: StoreDep,
    page: PageDep,
    symbol: Annotated[str | None, Query(description="Exact ticker, case-insensitive.")] = None,
    account_id: Annotated[str | None, Query(description="Exact account id.")] = None,
    status: Annotated[OrderStatus | None, Query(description="Exact order status.")] = None,
    side: Annotated[Side | None, Query(description="BUY or SELL.")] = None,
    min_quantity: Annotated[int | None, Query(ge=1)] = None,
    start: Annotated[datetime | None, Query(description="Inclusive lower bound on timestamp.")] = None,
    end: Annotated[datetime | None, Query(description="Inclusive upper bound on timestamp.")] = None,
) -> Page[Trade]:
    """Filter, then page.

    Filters compose as AND. Every filter is a vectorised boolean mask over the
    snapshot, so the cost is O(n) per request on a dataset already in memory -
    fast at this scale, and the first thing that has to change when the dataset
    stops fitting there (see `docs/system_design.md`).

    `total` reports matches *before* pagination, so a client can size a pager
    without walking every page.
    """
    frame = store.frame
    mask = frame["trade_id"].notna()  # all-True seed of the right length/index

    if symbol:
        mask &= frame["symbol"] == symbol.upper()
    if account_id:
        mask &= frame["account_id"] == account_id
    if status:
        mask &= frame["order_status"] == status.value
    if side:
        mask &= frame["side"] == side.value
    if min_quantity is not None:
        mask &= frame["quantity"] >= min_quantity
    if start is not None:
        mask &= frame["timestamp"] >= start
    if end is not None:
        mask &= frame["timestamp"] <= end

    matched = frame[mask]
    total = int(len(matched))
    window = matched.iloc[page.offset : page.offset + page.limit]

    return Page[Trade](
        items=[Trade.model_validate(row) for row in window.to_dict("records")],
        total=total,
        limit=page.limit,
        offset=page.offset,
        has_more=page.offset + page.limit < total,
    )


@router.get(
    "/{trade_id}",
    response_model=Trade,
    summary="Fetch one trade by id",
    responses={404: {"model": ErrorResponse, "description": "Unknown trade id"}},
)
def get_trade(
    store: StoreDep,
    trade_id: Annotated[str, Path(min_length=1, max_length=64)],
) -> Trade:
    """O(1) average lookup via the store's hash index, not a scan."""
    record = store.get_trade(trade_id)
    if record is None:
        raise TradeNotFoundError(trade_id)
    return Trade.model_validate(record)
