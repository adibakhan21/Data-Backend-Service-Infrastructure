"""Shared fixtures.

Every fixture is session-scoped where it is expensive and function-scoped where
a test might mutate it. The dataset is generated once (`seed=42`), so the whole
suite runs against identical, reproducible data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetrack.generator import GeneratorConfig, InjectionPlan, generate  # noqa: E402
from tradetrack.store import AnalyticsStore  # noqa: E402

# Small enough that the suite runs in seconds, large enough that per-symbol
# groups clear the detector's `failure_rate_min_samples` floor of 50.
TEST_ROWS = 20_000


@pytest.fixture(scope="session")
def dataset() -> tuple[pd.DataFrame, InjectionPlan]:
    """A fixed synthetic dataset plus its anomaly ground truth."""
    return generate(GeneratorConfig(n_rows=TEST_ROWS, seed=42))


@pytest.fixture(scope="session")
def frame(dataset) -> pd.DataFrame:
    return dataset[0]


@pytest.fixture(scope="session")
def plan(dataset) -> InjectionPlan:
    return dataset[1]


@pytest.fixture(scope="session")
def store(frame) -> AnalyticsStore:
    """A built store, shared across tests since nothing mutates it."""
    return AnalyticsStore.build(frame)


@pytest.fixture(scope="session")
def empty_frame() -> pd.DataFrame:
    """A schema-correct but empty frame, for the empty-input edge cases."""
    return pd.DataFrame(
        columns=[
            "trade_id", "timestamp", "symbol", "side", "quantity",
            "price", "order_status", "execution_latency_ms", "account_id",
        ]
    )


@pytest.fixture(scope="session")
def client(store):
    """TestClient bound to a store built from the fixture data.

    Used as a context manager so FastAPI's lifespan actually runs; without it
    `app.state.store` is never populated.
    """
    from fastapi.testclient import TestClient

    from tradetrack.api.main import create_app

    with TestClient(create_app(store=store)) as test_client:
        yield test_client


@pytest.fixture()
def empty_client(empty_frame):
    """TestClient over an empty dataset, for the 503 path."""
    from fastapi.testclient import TestClient

    from tradetrack.api.main import create_app

    with TestClient(create_app(store=AnalyticsStore.build(empty_frame))) as test_client:
        yield test_client
