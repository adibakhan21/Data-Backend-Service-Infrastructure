#!/usr/bin/env python3
"""Measure anomaly-detector precision/recall against injected ground truth.

The generator records exactly which trades and accounts it made anomalous, so
recall and precision here are real measurements, not estimates.

Protocol, and the reason it is worth reading: the quantity rule's threshold was
chosen by an F1 sweep, which means reporting F1 on that same seed would be
reporting a tuned-to-fit number. So the sweep runs on one seed and the frozen
threshold is then evaluated on seeds it has never seen. The held-out figures
are the ones worth quoting.

Usage:
    python scripts/evaluate_detector.py
    python scripts/evaluate_detector.py --rows 100000 --output benchmarks/detector_eval.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetrack.anomalies import (  # noqa: E402
    detect_failure_rate_anomalies,
    detect_latency_anomalies,
    detect_quantity_anomalies,
)
from tradetrack.config import Settings  # noqa: E402
from tradetrack.generator import GeneratorConfig, generate  # noqa: E402

TUNE_SEED = 42
HELDOUT_SEEDS = (7, 13, 101, 2024, 31337)


def score(predicted: set[str], actual: set[str]) -> tuple[float, float, float]:
    """Return (precision, recall, F1). Empty predictions score 0, not a crash."""
    true_positives = len(predicted & actual)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_seed(seed: int, rows: int, config: Settings) -> dict[str, tuple[float, float, float]]:
    """Run all three rules on one seed and score each against its ground truth."""
    frame, plan = generate(GeneratorConfig(n_rows=rows, seed=seed))
    return {
        "HIGH_LATENCY": score(
            {a.entity_id for a in detect_latency_anomalies(frame, config)},
            set(plan.latency_trade_ids),
        ),
        "UNUSUAL_QUANTITY": score(
            {a.entity_id for a in detect_quantity_anomalies(frame, config)},
            set(plan.oversized_trade_ids),
        ),
        "HIGH_FAILURE_RATE": score(
            {a.entity_id for a in detect_failure_rate_anomalies(frame, "account_id", config)},
            set(plan.failing_accounts),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/detector_eval.md"))
    args = parser.parse_args()

    lines: list[str] = [
        "# Anomaly detector evaluation",
        "",
        f"Dataset: {args.rows:,} synthetic trades per seed, ground truth from the generator's",
        "injection plan. Regenerate with `python scripts/evaluate_detector.py`.",
        "",
        "## Threshold sweep for UNUSUAL_QUANTITY (tuning seed only)",
        "",
        "Raw-scale vs log-scale MAD. The raw-scale rule cannot be fixed by moving k:",
        "it is flagging the lognormal's legitimate right tail at every threshold.",
        "",
        "| space | k | flags | precision | recall | F1 |",
        "|---|---|---|---|---|---|",
    ]

    frame, plan = generate(GeneratorConfig(n_rows=args.rows, seed=TUNE_SEED))
    truth = set(plan.oversized_trade_ids)

    best_k, best_f1 = None, -1.0
    for use_log, k in [(False, 5.0), (False, 10.0)] + [(True, k) for k in (3.0, 3.25, 3.5, 3.75, 4.0, 4.5)]:
        cfg = Settings(quantity_mad_threshold=k)
        if use_log:
            predicted = {a.entity_id for a in detect_quantity_anomalies(frame, cfg)}
        else:
            predicted = _raw_scale_quantity_flags(frame, k)
        precision, recall, f1 = score(predicted, truth)
        lines.append(
            f"| {'log' if use_log else 'raw'} | {k} | {len(predicted)} | "
            f"{precision:.3f} | {recall:.3f} | {f1:.3f} |"
        )
        if use_log and f1 > best_f1:
            best_k, best_f1 = k, f1

    lines += [
        "",
        f"Selected **k = {best_k}** (log space) by F1 on seed {TUNE_SEED}.",
        "",
        "## Held-out evaluation (threshold frozen)",
        "",
        f"Seeds {', '.join(str(s) for s in HELDOUT_SEEDS)} - never used for tuning.",
        "",
        "| rule | precision | recall | F1 |",
        "|---|---|---|---|",
    ]

    config = Settings(quantity_mad_threshold=best_k)
    per_rule: dict[str, list[tuple[float, float, float]]] = {}
    for seed in HELDOUT_SEEDS:
        for rule, result in evaluate_seed(seed, args.rows, config).items():
            per_rule.setdefault(rule, []).append(result)

    for rule, results in per_rule.items():
        precision = statistics.mean(r[0] for r in results)
        recall = statistics.mean(r[1] for r in results)
        f1 = statistics.mean(r[2] for r in results)
        lines.append(f"| {rule} | {precision:.3f} | {recall:.3f} | {f1:.3f} |")
        print(f"{rule:20} precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")

    lines += [
        "",
        "Means over the five held-out seeds. HIGH_FAILURE_RATE scores on a handful of",
        "accounts per seed, so its figures are coarse-grained by construction.",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"\nwrote {args.output}")
    return 0


def _raw_scale_quantity_flags(frame, k: float) -> set[str]:
    """The rejected raw-scale rule, kept solely so the comparison is reproducible."""
    from tradetrack.algorithms import median_absolute_deviation

    flagged: set[str] = set()
    for _, group in frame.groupby("symbol", sort=False):
        quantities = group["quantity"].astype(float).tolist()
        median, mad = median_absolute_deviation(quantities)
        if mad <= 0:
            continue
        threshold = median + k * mad * 1.4826
        flagged |= set(group.loc[group["quantity"] > threshold, "trade_id"])
    return flagged


if __name__ == "__main__":
    raise SystemExit(main())
