"""Triage graph, tools and structured outputs.

Every test here runs on the scripted backend: no API key, no network, fully
deterministic. That is the point of having a scripted backend at all - the graph
topology, the tool contract, the loop bound and the audit trail are all
properties of the system, not of whichever model is behind it, so they should be
testable without one.
"""

from __future__ import annotations

import pytest

from tradetrack.agents.graph import build_graph, triage
from tradetrack.agents.llm import ScriptedBackend, get_backend
from tradetrack.agents.schemas import (
    Classification,
    Decision,
    ToolChoice,
    TriageCategory,
    Verdict,
)
from tradetrack.agents.tools import TOOL_DESCRIPTIONS, build_tools, expected_first_tool


@pytest.fixture(scope="module")
def tools(store):
    return build_tools(store)


@pytest.fixture(scope="module")
def backend():
    return ScriptedBackend()


class TestTools:
    def test_every_tool_is_documented(self, tools):
        """A tool the model cannot see described is a tool it cannot use."""
        assert set(tools) == set(TOOL_DESCRIPTIONS)

    def test_symbol_metrics_returns_facts(self, tools):
        assert "AAPL" in tools["get_symbol_metrics"]("AAPL")

    def test_symbol_lookup_is_case_insensitive(self, tools):
        assert tools["get_symbol_metrics"]("aapl") == tools["get_symbol_metrics"]("AAPL")

    def test_account_history(self, tools, store):
        account = store.frame["account_id"].iloc[0]
        assert account in tools["get_account_history"](account)

    def test_trade_detail(self, tools, store):
        trade = store.frame["trade_id"].iloc[5]
        assert trade in tools["get_trade_detail"](trade)

    def test_global_baseline_ignores_its_argument(self, tools):
        assert tools["get_global_baseline"]("") == tools["get_global_baseline"]("anything")

    @pytest.mark.parametrize("tool_name,bad_input", [
        ("get_symbol_metrics", "NOT_A_SYMBOL"),
        ("get_account_history", "NOT_AN_ACCOUNT"),
        ("get_trade_detail", "NOT_A_TRADE"),
    ])
    def test_unknown_entity_reports_absence_rather_than_raising(self, tools, tool_name, bad_input):
        """A tool that raises ends the graph run. A tool that says 'no data'
        lets the agent reason about the absence and try something else."""
        assert "No " in tools[tool_name](bad_input)

    def test_whitespace_is_tolerated(self, tools):
        """Models routinely pad arguments. Stripping is cheaper than a retry."""
        assert tools["get_symbol_metrics"]("  AAPL  ") == tools["get_symbol_metrics"]("AAPL")


class TestRubric:
    @pytest.mark.parametrize("anomaly_type,scope,expected", [
        ("HIGH_FAILURE_RATE", "account", "get_account_history"),
        ("HIGH_FAILURE_RATE", "symbol", "get_global_baseline"),
        ("HIGH_LATENCY", "trade", "get_symbol_metrics"),
        ("UNUSUAL_QUANTITY", "trade", "get_symbol_metrics"),
    ])
    def test_expected_first_tool(self, anomaly_type, scope, expected):
        assert expected_first_tool(anomaly_type, scope) == expected

    def test_rubric_only_names_real_tools(self, tools):
        for anomaly_type in ("HIGH_FAILURE_RATE", "HIGH_LATENCY", "UNUSUAL_QUANTITY"):
            for scope in ("trade", "symbol", "account"):
                assert expected_first_tool(anomaly_type, scope) in tools


class TestScriptedBackend:
    def test_produces_each_schema(self, backend):
        prompt = "anomaly_type: HIGH_LATENCY\n  scope: trade\n  symbol: AAPL\n  entity_id: T1\n  severity: HIGH"
        assert isinstance(backend.structured(prompt, Classification), Classification)
        assert isinstance(backend.structured(prompt, ToolChoice), ToolChoice)
        assert isinstance(backend.structured(prompt, Decision), Decision)

    def test_rejects_an_unsupported_schema(self, backend):
        with pytest.raises(ValueError, match="cannot produce"):
            backend.structured("x", Verdict)  # type: ignore[arg-type]

    def test_mirrors_the_tool_rubric(self, backend):
        """The control arm has to answer the same question the prompt asks, or
        it is not a fair baseline."""
        prompt = "anomaly_type: HIGH_FAILURE_RATE\n  scope: account\n  entity_id: ACC00001\n  symbol: \n  severity: HIGH"
        assert backend.structured(prompt, ToolChoice).tool_name == "get_account_history"

    def test_latency_is_classified_as_infrastructure(self, backend):
        result = backend.structured("anomaly_type: HIGH_LATENCY", Classification)
        assert result.category is TriageCategory.INFRASTRUCTURE

    def test_auto_backend_without_a_key_falls_back(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert get_backend("auto").name == "scripted"

    def test_unknown_backend_name_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("gpt-9")


class TestGraph:
    def test_produces_a_complete_record(self, store, backend, tools):
        record = triage(store.anomalies[0], backend, tools)
        assert record.classification is not None
        assert record.decision is not None
        assert record.tool_calls and record.error is None

    def test_every_record_is_auditable(self, store, backend, tools):
        """The deliverable is the audit trail, not just the verdict."""
        for anomaly in store.anomalies[:5]:
            record = triage(anomaly, backend, tools)
            assert record.decision.reason and len(record.decision.reason) >= 15
            assert record.classification.rationale
            for call in record.tool_calls:
                assert call["why"] and "observation" in call

    def test_runs_every_anomaly_type(self, store, backend, tools):
        seen = set()
        for anomaly in store.anomalies:
            if anomaly.anomaly_type in seen:
                continue
            seen.add(anomaly.anomaly_type)
            assert triage(anomaly, backend, tools).decision is not None
        assert len(seen) >= 2

    def test_loop_is_bounded(self, store, tools):
        """A backend that always asks for more evidence must still terminate.
        Without the iteration bound this hangs forever."""

        class NeverSatisfied(ScriptedBackend):
            name = "never-satisfied"

            def structured(self, prompt, schema):
                if schema is Decision:
                    return Decision(
                        verdict=Verdict.NEED_MORE_EVIDENCE,
                        confidence=0.1,
                        reason="Insufficient evidence to reach any conclusion yet.",
                    )
                return super().structured(prompt, schema)

        record = triage(store.anomalies[0], NeverSatisfied(), tools, max_iterations=2)
        assert record.iterations <= 2

    def test_exhausted_budget_escalates_rather_than_closing(self, store, tools):
        """An anomaly nobody finished investigating belongs with a human."""

        class NeverSatisfied(ScriptedBackend):
            name = "never-satisfied"

            def structured(self, prompt, schema):
                if schema is Decision:
                    return Decision(
                        verdict=Verdict.NEED_MORE_EVIDENCE,
                        confidence=0.1,
                        reason="Insufficient evidence to reach any conclusion yet.",
                    )
                return super().structured(prompt, schema)

        record = triage(store.anomalies[0], NeverSatisfied(), tools, max_iterations=1)
        assert record.decision.verdict is Verdict.ESCALATE
        assert "budget exhausted" in record.decision.reason.lower()

    def test_hallucinated_tool_name_is_recorded_not_raised(self, store, tools):
        """The model must be told its call was invalid so the next iteration can
        correct it; crashing the run teaches it nothing."""

        class BadTool(ScriptedBackend):
            name = "bad-tool"

            def structured(self, prompt, schema):
                if schema is ToolChoice:
                    return ToolChoice(
                        tool_name="query_the_database",
                        argument="x",
                        why="This tool does not exist in the registry.",
                    )
                return super().structured(prompt, schema)

        record = triage(store.anomalies[0], BadTool(), tools)
        assert record.error is None
        assert record.tool_calls[0]["valid"] is False
        assert "no tool named" in record.tool_calls[0]["observation"]

    def test_backend_failure_does_not_crash_the_run(self, store, tools):
        """A provider outage should produce a record with an error field, not a
        traceback out of the graph."""

        class Broken(ScriptedBackend):
            name = "broken"

            def structured(self, prompt, schema):
                raise RuntimeError("provider unavailable")

        record = triage(store.anomalies[0], Broken(), tools)
        assert record.error is not None and "classify failed" in record.error

    def test_graph_compiles_with_expected_nodes(self, backend, tools):
        compiled = build_graph(backend, tools)
        nodes = set(compiled.get_graph().nodes)
        assert {"classify", "investigate", "execute_tool", "decide"} <= nodes
