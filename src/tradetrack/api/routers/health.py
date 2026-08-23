"""Liveness and snapshot introspection."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ... import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Service status plus what the loaded snapshot actually contains.

    `trades_loaded` and `snapshot_built_at` are here on purpose: "the process is
    up" and "the process can answer questions" are different states, and a
    health check that only reports the first one lets an empty deploy pass.
    """

    status: str
    version: str
    trades_loaded: int
    snapshot_built_at: datetime | None
    snapshot_build_seconds: float


@router.get("/health", response_model=HealthResponse, summary="Liveness and snapshot status")
def health(request: Request) -> HealthResponse:
    """Always 200 while the process is alive; the body says whether it is useful.

    Reads the store defensively. A health check that raises when startup has
    not finished is worse than useless: the orchestrator gets a 500, restarts
    the container, and the process never reaches the state where it would pass.
    """
    store = getattr(request.app.state, "store", None)
    if store is None:
        return HealthResponse(
            status="starting",
            version=__version__,
            trades_loaded=0,
            snapshot_built_at=None,
            snapshot_build_seconds=0.0,
        )
    return HealthResponse(
        status="ok" if not store.is_empty else "degraded",
        version=__version__,
        trades_loaded=0 if store.is_empty else len(store.frame),
        snapshot_built_at=store.loaded_at,
        snapshot_build_seconds=store.build_seconds,
    )
