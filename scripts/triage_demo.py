#!/usr/bin/env python3
"""Run a few anomalies through the triage agent and print the audit trail.

Prints every step - classification, each tool call with its justification and
the observation it returned, and the final verdict - because the audit trail is
the deliverable. A triage system that reaches good decisions without showing its
work cannot be used in an operations context.

Runs with no API key (falls back to the scripted rule-based backend), so this
always works:

    python scripts/triage_demo.py
    python scripts/triage_demo.py --backend groq --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetrack.agents.graph import triage  # noqa: E402
from tradetrack.agents.llm import get_backend  # noqa: E402
from tradetrack.agents.tools import build_tools  # noqa: E402
from tradetrack.generator import GeneratorConfig, generate  # noqa: E402
from tradetrack.store import AnalyticsStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="auto", choices=["auto", "groq", "scripted"])
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--limit", type=int, default=3, help="Anomalies to triage.")
    args = parser.parse_args()

    frame, _ = generate(GeneratorConfig(n_rows=args.rows, seed=42))
    store = AnalyticsStore.build(frame)
    tools = build_tools(store)
    backend = get_backend(args.backend)

    print(f"backend: {backend.name}")
    print(f"{len(store.anomalies)} anomalies detected; triaging {args.limit}\n")

    # One of each type where possible - the interesting differences are between
    # types, not between two instances of the same one.
    chosen, seen = [], set()
    for anomaly in store.anomalies:
        if anomaly.anomaly_type not in seen:
            seen.add(anomaly.anomaly_type)
            chosen.append(anomaly)
        if len(chosen) >= args.limit:
            break

    for anomaly in chosen:
        record = triage(anomaly, backend, tools)
        print("=" * 78)
        print(f"{record.anomaly_type}  {record.entity_id}  [{record.severity}]")
        print(f"  detector said : {anomaly.reason}")

        if record.error:
            print(f"  ERROR         : {record.error}\n")
            continue

        cls = record.classification
        print(f"  classified as : {cls.category.value} (urgency {cls.urgency}/5)")
        print(f"                  {cls.rationale}")
        for i, call in enumerate(record.tool_calls, 1):
            flag = "" if call["valid"] else "  [INVALID TOOL NAME]"
            print(f"  tool call {i}   : {call['tool_name']}('{call['argument']}'){flag}")
            print(f"                  why: {call['why']}")
            print(f"                  ->  {call['observation']}")
        decision = record.decision
        print(f"  VERDICT       : {decision.verdict.value} (confidence {decision.confidence:.2f})")
        print(f"                  {decision.reason}")
        print(f"  iterations    : {record.iterations}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
