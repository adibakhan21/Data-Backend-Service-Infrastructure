"""Application factory and process entrypoint.

`create_app()` is a factory rather than a module-level `app = FastAPI()` so the
test suite can build an app over a purpose-built store without ever touching
the developer's SQLite file or the environment's configuration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .. import __version__
from ..config import Settings, settings as default_settings
from ..store import AnalyticsStore
from .errors import register_error_handlers
from .routers import anomalies, health, metrics, signals, trades

DESCRIPTION = """
Analytics over synthetic trade/order logs: aggregate metrics, explainable
anomaly detection and rule-based signals.

**Signals are descriptive rules over synthetic data. They are not predictions,
they have not been backtested, and no profitability claim is made.**
"""


def create_app(store: AnalyticsStore | None = None, config: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        store: Pre-built store. When omitted, one is loaded from SQLite during
            startup - the production path. Tests pass one in directly.
        config: Settings override, mainly for tests.
    """
    cfg = config or default_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Loading in `lifespan` rather than at import time means the module can
        # be imported (by tests, by tooling) without a multi-second disk read,
        # and a load failure surfaces as a failed startup rather than a failed
        # import with a confusing traceback.
        app.state.store = store if store is not None else AnalyticsStore.from_sqlite(cfg.db_path, cfg)
        yield
        app.state.store = None

    app = FastAPI(
        title="TradeTrack Analytics Engine",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    register_error_handlers(app)
    for module in (health, trades, signals, anomalies, metrics):
        app.include_router(module.router)
    return app


app = create_app()
