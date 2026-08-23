"""Explainable anomaly detection: three rules, no model.

The deliberate choice here is *rules over ML*. A learned detector would give a
score; these give a sentence. When an operations team is paged at 3am, "account
ACC00174 failed 34.2% of 612 orders, 4.0 sigma above the 8.5% book-wide rate"
is actionable and auditable, and an isolation-forest score of 0.83 is not.

Every rule follows the same shape:
  baseline -> threshold -> compare -> severity from how far past the threshold.

Each rule also has a guard against its own most likely false positive, which is
documented inline. That guard is the part that matters: a detector that fires on
everything is the same as a detector that fires on nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .algorithms import median_absolute_deviation, percentile
from .config import Settings, settings as default_settings
from .models import FAILURE_STATUSES, Anomaly, Severity

# MAD is a consistent estimator of sigma for normal data only after scaling by
# this constant (1 / Phi^-1(0.75)). Without it, MAD-based z-scores are not
# comparable to ordinary sigma-based ones.
MAD_TO_SIGMA = 1.4826


def _severity_from_ratio(ratio: float) -> Severity:
    """Map "how many times past the threshold" onto a severity band.

    The bands are a presentation choice, not a statistical result: 1-2x over is
    LOW, 2-4x MEDIUM, 4-8x HIGH, beyond that CRITICAL.
    """
    if ratio < 2:
        return Severity.LOW
    if ratio < 4:
        return Severity.MEDIUM
    if ratio < 8:
        return Severity.HIGH
    return Severity.CRITICAL


def detect_latency_anomalies(
    frame: pd.DataFrame, config: Settings | None = None
) -> list[Anomaly]:
    """Flag trades whose execution latency is a genuine outlier.

    Baseline: the p99 of the whole dataset's latency.
    Threshold: max(p99, `latency_p99_floor_ms`).

    The floor is the false-positive guard. Every dataset has a p99 by
    definition, so a bare "above p99" rule always flags exactly 1% of trades
    even when the venue is perfectly healthy. The absolute floor means a
    uniformly fast dataset produces zero alerts, which is the correct answer.

    Complexity:
        Time  O(n log n) - the percentile sort; the scan itself is O(n).
        Space O(n)
    """
    cfg = config or default_settings
    if frame.empty:
        return []

    latencies = frame["execution_latency_ms"].to_numpy()
    p99 = percentile(latencies.tolist(), 99)
    threshold = max(p99, cfg.latency_p99_floor_ms)

    flagged = frame[frame["execution_latency_ms"] > threshold]
    anomalies: list[Anomaly] = []
    for row in flagged.itertuples(index=False):
        value = float(row.execution_latency_ms)
        anomalies.append(
            Anomaly(
                anomaly_type="HIGH_LATENCY",
                severity=_severity_from_ratio(value / threshold),
                scope="trade",
                entity_id=str(row.trade_id),
                symbol=str(row.symbol),
                reason=(
                    f"Execution latency {value:.1f}ms is {value / threshold:.1f}x the alert "
                    f"threshold of {threshold:.1f}ms (dataset p99 {p99:.1f}ms, floor "
                    f"{cfg.latency_p99_floor_ms:.0f}ms)."
                ),
                metric_name="execution_latency_ms",
                metric_value=round(value, 2),
                threshold=round(threshold, 2),
            )
        )
    return anomalies


def detect_failure_rate_anomalies(
    frame: pd.DataFrame, group_by: str = "account_id", config: Settings | None = None
) -> list[Anomaly]:
    """Flag accounts (or symbols) failing far more often than the book.

    Baseline: the global failure rate p.
    Threshold: p + sigma * SE, where SE = sqrt(p(1-p)/n) is the *binomial
    standard error for that group's own sample size*.

    Using a per-group standard error rather than one fixed percentage is the
    false-positive guard, and it is the statistically correct move: an account
    with 60 orders has a much wider natural spread than one with 6,000, so a
    flat "flag anything above 20%" rule drowns in small-account noise. The
    `failure_rate_min_samples` floor discards groups too small for even the SE
    to be meaningful.

    Complexity:
        Time  O(n) - one vectorised groupby pass.
        Space O(g) for g groups.
    """
    cfg = config or default_settings
    if frame.empty:
        return []
    if group_by not in {"account_id", "symbol"}:
        raise ValueError("group_by must be 'account_id' or 'symbol'")

    work = frame.copy()
    work["is_failure"] = work["order_status"].isin(FAILURE_STATUSES)
    global_rate = float(work["is_failure"].mean())

    grouped = work.groupby(group_by, sort=False)["is_failure"].agg(["size", "sum"])
    grouped = grouped[grouped["size"] >= cfg.failure_rate_min_samples]

    anomalies: list[Anomaly] = []
    for entity, row in grouped.iterrows():
        n = int(row["size"])
        rate = float(row["sum"]) / n
        standard_error = math.sqrt(max(global_rate * (1 - global_rate), 1e-12) / n)
        threshold = global_rate + cfg.failure_rate_sigma * standard_error
        if rate <= threshold:
            continue

        sigmas = (rate - global_rate) / standard_error
        anomalies.append(
            Anomaly(
                anomaly_type="HIGH_FAILURE_RATE",
                severity=_severity_from_ratio(sigmas / cfg.failure_rate_sigma),
                scope="account" if group_by == "account_id" else "symbol",
                entity_id=str(entity),
                symbol=str(entity) if group_by == "symbol" else None,
                reason=(
                    f"{entity} failed {rate:.1%} of {n} orders, {sigmas:.1f} sigma above the "
                    f"book-wide rate of {global_rate:.1%} (alert threshold {threshold:.1%})."
                ),
                metric_name="failure_rate",
                metric_value=round(rate, 6),
                threshold=round(threshold, 6),
            )
        )
    return anomalies


def detect_quantity_anomalies(
    frame: pd.DataFrame, config: Settings | None = None
) -> list[Anomaly]:
    """Flag orders far larger than that symbol's normal size.

    Works in **log space**, and that is the whole design.

    Order sizes are lognormal, so on the raw scale their right tail is
    enormous and perfectly normal. A median + k*MAD rule on raw quantities
    therefore flags the entire natural tail: measured on 50k trades with 78
    deliberately-injected block orders, the raw-scale rule at k=5 returned
    2,267 alerts - 100% recall at **3.4% precision**. An alerting system with
    97 false alarms per true one is worse than no alerting system, because the
    team stops reading it.

    Taking the log first turns the multiplicative tail into an additive one, so
    a symmetric robust z-score becomes valid. The same rule on the same data at
    k=3.5 gives **86.4% recall at 90.9% precision** (mean over five held-out
    seeds; see `scripts/evaluate_detector.py`).

    Baseline: median of log(quantity) for that symbol.
    Threshold: median + k * MAD * 1.4826, all in log space.

    Symbols whose MAD is zero (every order identical) are skipped: the
    threshold would collapse onto the median and flag every larger order.

    Complexity:
        Time  O(n log n) - two medians per symbol.
        Space O(n)
    """
    cfg = config or default_settings
    if frame.empty:
        return []

    anomalies: list[Anomaly] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        quantities = group["quantity"].astype(float).to_numpy()
        # quantity is validated > 0 by the Trade model, so log is always safe.
        log_quantities: Sequence[float] = np.log(quantities).tolist()
        log_median, log_mad = median_absolute_deviation(log_quantities)
        if log_mad <= 0:
            continue

        scaled = log_mad * MAD_TO_SIGMA
        log_threshold = log_median + cfg.quantity_mad_threshold * scaled
        threshold = float(np.exp(log_threshold))

        mask = quantities > threshold
        for row in group[mask].itertuples(index=False):
            value = float(row.quantity)
            robust_z = (math.log(value) - log_median) / scaled
            anomalies.append(
                Anomaly(
                    anomaly_type="UNUSUAL_QUANTITY",
                    severity=_severity_from_ratio(robust_z / cfg.quantity_mad_threshold),
                    scope="trade",
                    entity_id=str(row.trade_id),
                    symbol=str(symbol),
                    reason=(
                        f"Order size {int(value)} on {symbol} is {value / threshold:.1f}x the "
                        f"alert threshold of {threshold:.0f} (robust z-score {robust_z:.1f} in "
                        f"log space; typical size {math.exp(log_median):.0f})."
                    ),
                    metric_name="quantity",
                    metric_value=value,
                    threshold=round(threshold, 2),
                )
            )
    return anomalies


SEVERITY_ORDER: dict[str, int] = {
    Severity.CRITICAL.value: 3,
    Severity.HIGH.value: 2,
    Severity.MEDIUM.value: 1,
    Severity.LOW.value: 0,
}


def detect_all(frame: pd.DataFrame, config: Settings | None = None) -> list[Anomaly]:
    """Run every rule and return findings sorted by severity, then magnitude.

    Complexity:
        Time  O(n log n), dominated by the per-symbol medians.
        Space O(n)
    """
    findings = (
        detect_latency_anomalies(frame, config)
        + detect_failure_rate_anomalies(frame, "account_id", config)
        + detect_failure_rate_anomalies(frame, "symbol", config)
        + detect_quantity_anomalies(frame, config)
    )
    findings.sort(
        key=lambda a: (SEVERITY_ORDER[a.severity.value], a.metric_value / max(a.threshold, 1e-9)),
        reverse=True,
    )
    return findings
