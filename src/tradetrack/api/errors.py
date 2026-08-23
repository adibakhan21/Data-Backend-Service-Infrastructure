"""Error types and the handlers that turn them into HTTP responses.

Everything the API can fail with is normalised into one JSON shape:

    {"error": "<machine-readable code>", "detail": "<human explanation>"}

so a client only ever writes one error path. The alternative - FastAPI's
default `{"detail": ...}` for HTTPException next to a bespoke shape for
everything else - forces callers to branch on status code to find the message.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """The single error envelope, published in the OpenAPI schema."""

    error: str
    detail: str


class TradeNotFoundError(Exception):
    """Raised when a trade_id is not present in the current snapshot."""

    def __init__(self, trade_id: str) -> None:
        self.trade_id = trade_id
        super().__init__(trade_id)


class SymbolNotFoundError(Exception):
    """Raised when a symbol has no trades in the current snapshot."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        super().__init__(symbol)


class DatasetEmptyError(Exception):
    """Raised when the store holds no data.

    503 rather than 500: the service is fine, the data has not been loaded yet.
    That distinction is what stops an unseeded deploy from paging someone about
    a bug that does not exist.
    """


def register_error_handlers(app: FastAPI) -> None:
    """Attach every handler to the app."""

    @app.exception_handler(TradeNotFoundError)
    async def _trade_not_found(_: Request, exc: TradeNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                error="trade_not_found",
                detail=f"No trade with id '{exc.trade_id}' in the current snapshot.",
            ).model_dump(),
        )

    @app.exception_handler(SymbolNotFoundError)
    async def _symbol_not_found(_: Request, exc: SymbolNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                error="symbol_not_found",
                detail=f"No trades for symbol '{exc.symbol}' in the current snapshot.",
            ).model_dump(),
        )

    @app.exception_handler(DatasetEmptyError)
    async def _dataset_empty(_: Request, __: DatasetEmptyError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="dataset_empty",
                detail="No trades loaded. Run `python scripts/generate_data.py` first.",
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_failed(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        location = ".".join(str(p) for p in first["loc"][1:]) or "request"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=ErrorResponse(
                error="validation_error", detail=f"{location}: {first['msg']}"
            ).model_dump(),
        )
