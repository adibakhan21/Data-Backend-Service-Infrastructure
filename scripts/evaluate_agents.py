#!/usr/bin/env python3
"""Score the triage agent against ground truth.

Three things are measured, and each has a real answer key rather than a
self-assessment:

1. **Tool-selection accuracy** - did the agent's FIRST tool call match the one a
   competent analyst would make (`tools.expected_first_tool`)? Only the first
   call is judged: later calls legitimately depend on what earlier ones
   returned, so scoring them would penalise correct adaptive behaviour.

2. **Structured-output validity** - what fraction of LLM calls returned an
   object that validated against its Pydantic schema on the first attempt? This
   is the number that decides whether an agent is deployable at all.

3. **Routing accuracy** - the generator records which entities it deliberately
   made anomalous. A finding on an injected entity SHOULD escalate; a finding on
   a clean entity is a detector false positive and SHOULD auto-close. That gives
   a genuine label, not a vibe.

The scripted backend runs the same cases as a control. An agent that only
matches a handful of if-statements has not earned its latency, and reporting the
model's score without that baseline would hide it.

Usage:
    export GROQ_API_KEY=...
    python scripts/evaluate_agents.py --backend groq --cases 30
    python scripts/evaluate_agents.py --backend scripted   # no key needed
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetrack.agents.graph import triage  # noqa: E402
from tradetrack.agents.llm import get_backend  # noqa: E402
from tradetrack.agents.schemas import Verdict  # noqa: E402
from tradetrack.agents.tools import build_tools, expected_first_tool  # noqa: E402
from tradetrack.generator import GeneratorConfig, generate  # noqa: E402
from tradetrack.store import AnalyticsStore  # noqa: E402

# LLM calls per anomaly when no re-investigation happens: classify, investigate,
# decide. Used to compute validity as a rate rather than a raw count.
CALLS_PER_RUN = 3


def stratified_cases(store: AnalyticsStore, limit: int) -> list:
    """Take a balanced sample across anomaly types.

    Unstratified sampling would be dominated by HIGH_LATENCY, which is by far
    the most common finding - and the tool-selection rubric differs by type, so
    an unbalanced sample would mostly measure one branch.
    """
    by_type: dict[str, list] = {}
    for anomaly in store.anomalies:
        by_type.setdefault(anomaly.anomaly_type, []).append(anomaly)

    per_type = max(1, limit // max(len(by_type), 1))
    sample: list = []
    for anomalies in by_type.values():
        sample.extend(anomalies[:per_type])
    return sample[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="auto", choices=["auto", "groq", "scripted"])
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/agent_eval.md"))
    args = parser.parse_args()

    frame, plan = generate(GeneratorConfig(n_rows=args.rows, seed=args.seed))
    store = AnalyticsStore.build(frame)
    tools = build_tools(store)
    backend = get_backend(args.backend)

    # Ground truth: everything the generator deliberately made anomalous.
    injected = set(plan.latency_trade_ids) | set(plan.oversized_trade_ids) | set(plan.failing_accounts)

    cases = stratified_cases(store, args.cases)
    print(f"backend={backend.name}  cases={len(cases)}  rows={args.rows:,}\n")

    tool_hits = tool_total = 0
    route_hits = route_total = 0
    valid_calls = total_calls = 0
    invalid_tool_names = 0
    verdicts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    started = time.perf_counter()

    for i, anomaly in enumerate(cases, 1):
        record = triage(anomaly, backend, tools)

        # (2) structured-output validity
        produced = sum(x is not None for x in (record.classification, record.decision))
        produced += len(record.tool_calls)
        total_calls += CALLS_PER_RUN + max(0, record.iterations - 1) * 2
        valid_calls += produced

        # (1) tool selection, first call only
        if record.tool_calls:
            expected = expected_first_tool(anomaly.anomaly_type, anomaly.scope)
            chosen = record.tool_calls[0]["tool_name"]
            tool_total += 1
            tool_hits += int(chosen == expected)
            invalid_tool_names += int(not record.tool_calls[0]["valid"])

        # (3) routing
        if record.decision:
            verdicts[record.decision.verdict.value] += 1
            should_escalate = anomaly.entity_id in injected
            did_escalate = record.decision.verdict is Verdict.ESCALATE
            route_total += 1
            route_hits += int(should_escalate == did_escalate)

        if record.classification:
            categories[record.classification.category.value] += 1

        status = "ok" if not record.error else f"ERR {record.error[:40]}"
        print(f"  [{i:>3}/{len(cases)}] {anomaly.anomaly_type:<18} {status}", flush=True)

    elapsed = time.perf_counter() - started

    tool_accuracy = tool_hits / tool_total if tool_total else 0.0
    route_accuracy = route_hits / route_total if route_total else 0.0
    validity = valid_calls / total_calls if total_calls else 0.0

    lines = [
        "# Agent triage evaluation",
        "",
        f"Backend: `{backend.name}` · {len(cases)} anomalies sampled from {args.rows:,} trades ",
        f"(seed {args.seed}, stratified by anomaly type). Regenerate with ",
        "`python scripts/evaluate_agents.py --backend groq`.",
        "",
        "| Metric | Result | Answer key |",
        "|---|---|---|",
        f"| Tool-selection accuracy (first call) | **{tool_accuracy:.1%}** | `tools.expected_first_tool` rubric |",
        f"| Structured-output validity | **{validity:.1%}** | Pydantic schema validation, first attempt |",
        f"| Routing accuracy | **{route_accuracy:.1%}** | generator's anomaly-injection ground truth |",
        f"| Hallucinated tool names | {invalid_tool_names} of {tool_total} | name not in the tool registry |",
        f"| Wall clock | {elapsed:.1f}s ({elapsed / max(len(cases),1):.2f}s per anomaly) | |",
        "",
        f"Verdicts: {dict(verdicts)}  ",
        f"Categories: {dict(categories)}",
        "",
        "## How to read this",
        "",
        "Tool selection is judged on the FIRST call only. Later calls depend on what",
        "earlier ones returned, so grading them against a fixed key would punish the",
        "agent for adapting correctly.",
        "",
        "Routing accuracy is bounded above by the detector: a finding the detector",
        "should never have raised can still be routed correctly (auto-closed), but the",
        "agent cannot escalate something that was never surfaced to it.",
        "",
        "Run `--backend scripted` for the rule-based control arm. The model has to beat",
        "a handful of if-statements to justify its latency.",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))

    print(f"\n  tool-selection accuracy : {tool_accuracy:.1%}")
    print(f"  structured-output valid : {validity:.1%}")
    print(f"  routing accuracy        : {route_accuracy:.1%}")
    print(f"  hallucinated tool names : {invalid_tool_names}/{tool_total}")
    print(f"  elapsed                 : {elapsed:.1f}s")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
