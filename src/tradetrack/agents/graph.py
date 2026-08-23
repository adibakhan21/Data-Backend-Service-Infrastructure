"""The LangGraph triage state machine.

    classify -> investigate -> execute_tool -> decide -> [route | investigate]
                     ^                                        |
                     +----------------------------------------+
                          bounded re-investigation

The cycle is the reason this is a graph and not a function call. `decide` may
return NEED_MORE_EVIDENCE, sending control back to `investigate` for another
tool call with the previous evidence in context. That loop is bounded by
`max_iterations`; when the budget runs out the run terminates with an explicit
ESCALATE rather than looping or silently returning a low-confidence guess.
Escalating on exhaustion is the safe default: an un-investigated anomaly should
reach a human, not a wastebasket.

Every node appends to the record rather than overwriting it, so the finished
`TriageRecord` is a complete audit trail.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from ..models import Anomaly
from .llm import LLMBackend
from .schemas import Classification, Decision, ToolChoice, TriageRecord, Verdict
from .tools import TOOL_DESCRIPTIONS

MAX_ITERATIONS = 3


def _append(left: list, right: list) -> list:
    """Reducer so concurrent-safe list state accumulates instead of replacing."""
    return left + right


class TriageState(TypedDict, total=False):
    """State carried through the graph.

    `evidence` uses an append reducer because each loop iteration must add to
    what previous iterations found - replacing it would make the whole cycle
    pointless, since the model would re-decide on a single fact each time.
    """

    anomaly: dict
    classification: dict
    pending_tool: dict
    evidence: Annotated[list, _append]
    tool_calls: Annotated[list, _append]
    decision: dict
    iterations: int
    error: str


def _anomaly_block(anomaly: dict) -> str:
    """The anomaly rendered as prompt fields.

    Rendered as `key: value` lines rather than JSON on purpose: it keeps the
    scripted backend's field extraction simple, and small models follow flat
    key-value context more reliably than nested objects.
    """
    return (
        f"  anomaly_type: {anomaly['anomaly_type']}\n"
        f"  severity: {anomaly['severity']}\n"
        f"  scope: {anomaly['scope']}\n"
        f"  entity_id: {anomaly['entity_id']}\n"
        f"  symbol: {anomaly.get('symbol') or ''}\n"
        f"  metric: {anomaly['metric_name']} = {anomaly['metric_value']} "
        f"(threshold {anomaly['threshold']})\n"
        f"  detector_reason: {anomaly['reason']}"
    )


def build_graph(backend: LLMBackend, tools: dict, max_iterations: int = MAX_ITERATIONS):
    """Compile the triage graph for one backend and tool set."""

    tool_menu = "\n".join(f"  - {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())

    # ---- nodes ----------------------------------------------------------

    def classify(state: TriageState) -> dict:
        """Assign a probable cause category and an urgency."""
        prompt = (
            "You are triaging an anomaly raised by a trade-analytics system.\n\n"
            f"ANOMALY:\n{_anomaly_block(state['anomaly'])}\n\n"
            "Classify the most likely CAUSE:\n"
            "  INFRASTRUCTURE   - venue/network/system slowness (latency outliers)\n"
            "  CLIENT_BEHAVIOUR - one account behaving unusually (failure rates by account)\n"
            "  MARKET_ACTIVITY  - legitimate large or unusual flow (outsized orders)\n"
            "  DATA_QUALITY     - the record itself looks wrong\n\n"
            "Also give urgency 1-5 (5 = page someone now) and a one-sentence rationale."
        )
        try:
            result = backend.structured(prompt, Classification)
            return {"classification": result.model_dump(mode="json"), "iterations": 0}
        except Exception as exc:  # a bad model response must not kill the run
            return {"error": f"classify failed: {exc}", "iterations": 0}

    def investigate(state: TriageState) -> dict:
        """Choose the next tool call."""
        if state.get("error"):
            return {}
        gathered = state.get("evidence", [])
        history = (
            "\n\nEVIDENCE SO FAR:\n" + "\n".join(f"  - {e}" for e in gathered)
            if gathered
            else "\n\nEVIDENCE SO FAR: none - this is the first call."
        )
        prompt = (
            "You are investigating an anomaly. Choose ONE tool call that will best "
            "establish whether it is a real problem or explainable.\n\n"
            f"ANOMALY:\n{_anomaly_block(state['anomaly'])}"
            f"{history}\n\n"
            f"AVAILABLE TOOLS:\n{tool_menu}\n\n"
            "An anomaly is a value that looks extreme. To judge it you need a BASELINE "
            "to compare against, not a restatement of the value itself - the anomaly "
            "block above already gives you the observed value and the threshold it "
            "crossed. Prefer the tool that establishes the relevant baseline.\n\n"
            "Return the exact tool name, its single string argument, and what you "
            "expect the call to establish. Do not repeat a call you already made."
        )
        try:
            choice = backend.structured(prompt, ToolChoice)
            return {"pending_tool": choice.model_dump(mode="json")}
        except Exception as exc:
            return {"error": f"investigate failed: {exc}"}

    def execute_tool(state: TriageState) -> dict:
        """Run the chosen tool.

        A hallucinated tool name is recorded as evidence rather than raised:
        telling the model its call was invalid lets the next iteration correct
        itself, and it keeps the failure visible in the audit trail.
        """
        if state.get("error") or not state.get("pending_tool"):
            return {"iterations": state.get("iterations", 0) + 1}

        choice = state["pending_tool"]
        name, argument = choice["tool_name"], choice["argument"]
        tool = tools.get(name)

        if tool is None:
            observation = f"ERROR: no tool named '{name}'. Valid tools: {', '.join(tools)}."
            ok = False
        else:
            try:
                observation = tool(argument)
                ok = True
            except Exception as exc:
                observation = f"ERROR: {name}('{argument}') raised {type(exc).__name__}: {exc}"
                ok = False

        return {
            "evidence": [f"{name}('{argument}') -> {observation}"],
            "tool_calls": [{"tool_name": name, "argument": argument, "why": choice["why"],
                            "observation": observation, "valid": ok}],
            "iterations": state.get("iterations", 0) + 1,
        }

    def decide(state: TriageState) -> dict:
        """Judge the evidence, or ask for more."""
        if state.get("error"):
            return {}
        budget_left = max_iterations - state.get("iterations", 0)
        prompt = (
            "Decide what should happen to this anomaly.\n\n"
            f"ANOMALY:\n{_anomaly_block(state['anomaly'])}\n\n"
            f"CLASSIFICATION: {state.get('classification', {})}\n\n"
            "EVIDENCE:\n" + "\n".join(f"  - {e}" for e in state.get("evidence", [])) + "\n\n"
            "Choose one verdict:\n"
            "  AUTO_CLOSE          - the evidence explains it; no human needed\n"
            "  ESCALATE            - a human should look at this\n"
            "  NEED_MORE_EVIDENCE  - one more tool call would settle it\n"
            f"(You may request more evidence {budget_left} more time(s).)\n\n"
            "Give a confidence 0-1 and the justification a human will read."
        )
        try:
            result = backend.structured(prompt, Decision)
            return {"decision": result.model_dump(mode="json")}
        except Exception as exc:
            return {"error": f"decide failed: {exc}"}

    # ---- edges ----------------------------------------------------------

    def route(state: TriageState) -> str:
        """Loop back only if the model asked AND budget remains."""
        if state.get("error"):
            return "end"
        decision = state.get("decision") or {}
        if decision.get("verdict") != Verdict.NEED_MORE_EVIDENCE.value:
            return "end"
        return "investigate" if state.get("iterations", 0) < max_iterations else "end"

    graph = StateGraph(TriageState)
    graph.add_node("classify", classify)
    graph.add_node("investigate", investigate)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("decide", decide)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "investigate")
    graph.add_edge("investigate", "execute_tool")
    graph.add_edge("execute_tool", "decide")
    graph.add_conditional_edges("decide", route, {"investigate": "investigate", "end": END})

    return graph.compile()


def triage(anomaly: Anomaly, backend: LLMBackend, tools: dict,
           max_iterations: int = MAX_ITERATIONS) -> TriageRecord:
    """Run one anomaly through the graph and return its audit record."""
    compiled = build_graph(backend, tools, max_iterations)
    final = compiled.invoke({
        "anomaly": anomaly.model_dump(mode="json"),
        "evidence": [], "tool_calls": [], "iterations": 0,
    })

    decision = final.get("decision")
    # Budget exhaustion leaves NEED_MORE_EVIDENCE as the last verdict. Convert
    # it to ESCALATE: an anomaly nobody finished investigating belongs with a
    # human, not closed by default.
    if decision and decision.get("verdict") == Verdict.NEED_MORE_EVIDENCE.value:
        decision = {**decision, "verdict": Verdict.ESCALATE.value,
                    "reason": decision["reason"] + " [Investigation budget exhausted; escalated.]"}

    return TriageRecord(
        anomaly_type=anomaly.anomaly_type,
        entity_id=anomaly.entity_id,
        severity=anomaly.severity.value,
        classification=Classification.model_validate(final["classification"])
        if final.get("classification") else None,
        tool_calls=final.get("tool_calls", []),
        decision=Decision.model_validate(decision) if decision else None,
        iterations=final.get("iterations", 0),
        error=final.get("error"),
    )
