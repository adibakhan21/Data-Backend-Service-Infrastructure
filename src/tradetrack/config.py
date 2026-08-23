"""Runtime configuration, read from environment variables.

Every value has a working default so the project runs with zero setup.
Override any of them with a `TRADETRACK_`-prefixed environment variable,
e.g. `TRADETRACK_DB_PATH=/tmp/other.db`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings.

    Attributes:
        db_path: SQLite file holding generated trades.
        api_page_size_default: Default number of rows returned by list endpoints.
        api_page_size_max: Hard cap on ``limit`` so a client cannot ask for the
            whole table in one request.
        latency_p99_floor_ms: Absolute floor for the high-latency rule. Without
            it, a dataset with uniformly fast trades still flags its own top 1%.
        failure_rate_min_samples: Minimum trades before a per-entity failure
            rate is trusted. Prevents "1 of 1 failed = 100%" noise.
        failure_rate_sigma: How many standard deviations above the global
            failure rate an entity must sit to be flagged.
        quantity_mad_threshold: Robust z-score cutoff for the unusual-quantity
            rule, applied in log space. Default 3.5 was selected by an F1
            sweep on one seed and validated on five held-out seeds - see
            `scripts/evaluate_detector.py`.
        signal_window: Number of most recent trades per symbol used by the
            momentum/volume signal rules.
    """

    model_config = SettingsConfigDict(env_prefix="TRADETRACK_", env_file=".env", extra="ignore")

    db_path: Path = PROJECT_ROOT / "data" / "trades.db"

    api_page_size_default: int = 100
    api_page_size_max: int = 1000

    latency_p99_floor_ms: float = 250.0
    failure_rate_min_samples: int = 50
    failure_rate_sigma: float = 3.0
    quantity_mad_threshold: float = 3.5
    signal_window: int = 200


settings = Settings()
