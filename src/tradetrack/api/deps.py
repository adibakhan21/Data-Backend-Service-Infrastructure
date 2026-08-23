"""Shared dependencies: the store handle and the pagination contract.

The store lives on `app.state`, not in a module-level global. That is what lets
`tests/test_api.py` build an app over a tiny in-memory dataset without touching
the developer's SQLite file, and what would let a future version swap in a
different backing store without editing a single router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Query, Request

from ..config import settings
from ..store import AnalyticsStore
from .errors import DatasetEmptyError


def get_store(request: Request) -> AnalyticsStore:
    """Return the app's store, or 503 if nothing has been loaded."""
    store: AnalyticsStore | None = getattr(request.app.state, "store", None)
    if store is None or store.is_empty:
        raise DatasetEmptyError()
    return store


@dataclass
class Pagination:
    """Validated `limit`/`offset` pair.

    `limit` is capped at `api_page_size_max` by the query constraint itself, so
    a client cannot request the entire table in one response and turn a cheap
    endpoint into an out-of-memory event. The cap is enforced by FastAPI before
    any handler code runs, which means it cannot be forgotten in a new route.
    """

    limit: int
    offset: int


def get_pagination(
    limit: Annotated[
        int,
        Query(ge=1, le=settings.api_page_size_max, description="Maximum rows to return."),
    ] = settings.api_page_size_default,
    offset: Annotated[int, Query(ge=0, description="Rows to skip.")] = 0,
) -> Pagination:
    """Dependency yielding validated pagination parameters."""
    return Pagination(limit=limit, offset=offset)


StoreDep = Annotated[AnalyticsStore, Depends(get_store)]
PageDep = Annotated[Pagination, Depends(get_pagination)]
