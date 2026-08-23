"""LLM backends.

Two implementations behind one interface:

* `GroqBackend` - a real model (llama-3.3-70b) via langchain-groq, using
  LangChain's `with_structured_output` so the provider enforces the schema.
* `ScriptedBackend` - deterministic, no network, no key.

The scripted backend is not a mock in the testing sense. It is a rule-based
baseline that makes the *same decisions the prompt asks for*, which serves two
purposes: the graph, the tools and the audit trail stay fully testable in CI
with no API key, and the evaluation gets a control arm. "The agent chose the
right tool 91% of the time" means little until you know what a trivial rule
scores on the same cases.
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .schemas import Classification, Decision, ToolChoice, TriageCategory, Verdict

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMBackend(Protocol):
    """Anything that can turn a prompt into a validated Pydantic object."""

    name: str

    def structured(self, prompt: str, schema: type[T]) -> T:
        """Return an instance of `schema`, or raise if the model cannot produce one."""
        ...


class GroqBackend:
    """Groq-hosted model with provider-enforced structured output."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> None:
        from langchain_groq import ChatGroq

        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys, "
                "then `export GROQ_API_KEY=...`, or use the scripted backend."
            )
        # temperature=0 because this is a decisioning task, not a generative one.
        # Sampling would make the evaluation numbers irreproducible for no gain.
        self._client = ChatGroq(model=model, temperature=temperature)
        self.name = f"groq:{model}"

    def structured(self, prompt: str, schema: type[T]) -> T:
        return self._client.with_structured_output(schema).invoke(prompt)


class ScriptedBackend:
    """Deterministic rule-based baseline. No network, no key.

    Every branch below mirrors an instruction that appears in the corresponding
    prompt, so this is a fair control: it is what you get from encoding the
    prompt's rules directly instead of asking a model to follow them.
    """

    name = "scripted"

    def structured(self, prompt: str, schema: type[T]) -> T:
        if schema is Classification:
            return self._classify(prompt)  # type: ignore[return-value]
        if schema is ToolChoice:
            return self._choose_tool(prompt)  # type: ignore[return-value]
        if schema is Decision:
            return self._decide(prompt)  # type: ignore[return-value]
        raise ValueError(f"ScriptedBackend cannot produce {schema.__name__}")

    @staticmethod
    def _classify(prompt: str) -> Classification:
        if "HIGH_LATENCY" in prompt:
            return Classification(
                category=TriageCategory.INFRASTRUCTURE,
                urgency=4,
                rationale="Latency outliers point at venue or network degradation rather than client intent.",
            )
        if "HIGH_FAILURE_RATE" in prompt:
            return Classification(
                category=TriageCategory.CLIENT_BEHAVIOUR,
                urgency=4,
                rationale="A failure rate concentrated in one entity suggests that entity's own behaviour.",
            )
        return Classification(
            category=TriageCategory.MARKET_ACTIVITY,
            urgency=2,
            rationale="An outsized order is most often legitimate block activity.",
        )

    @staticmethod
    def _choose_tool(prompt: str) -> ToolChoice:
        # Mirrors tools.expected_first_tool, reading the anomaly fields out of
        # the prompt the graph built.
        if "HIGH_FAILURE_RATE" in prompt and "scope: account" in prompt:
            entity = ScriptedBackend._field(prompt, "entity_id")
            return ToolChoice(
                tool_name="get_account_history",
                argument=entity,
                why="Establish whether this account fails consistently or had one bad session.",
            )
        if "HIGH_FAILURE_RATE" in prompt:
            return ToolChoice(
                tool_name="get_global_baseline",
                argument="",
                why="Compare the symbol's failure rate against the book-wide rate.",
            )
        symbol = ScriptedBackend._field(prompt, "symbol")
        return ToolChoice(
            tool_name="get_symbol_metrics",
            argument=symbol,
            why="Establish the symbol's own baseline before judging this trade.",
        )

    @staticmethod
    def _decide(prompt: str) -> Decision:
        if "CRITICAL" in prompt or "HIGH" in prompt:
            return Decision(
                verdict=Verdict.ESCALATE,
                confidence=0.8,
                reason="Severity is high and the gathered evidence does not explain the deviation away.",
            )
        return Decision(
            verdict=Verdict.AUTO_CLOSE,
            confidence=0.7,
            reason="Low severity and the entity's baseline accounts for the observed value.",
        )

    @staticmethod
    def _field(prompt: str, key: str) -> str:
        """Pull `key: value` out of the prompt block the graph builds."""
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{key}:"):
                return stripped.split(":", 1)[1].strip()
        return ""


def get_backend(name: str = "auto") -> LLMBackend:
    """Resolve a backend by name.

    "auto" prefers Groq when a key is present and falls back to scripted, so the
    same command works with or without credentials rather than failing at the
    first import.
    """
    if name == "scripted":
        return ScriptedBackend()
    if name == "groq":
        return GroqBackend()
    if name == "auto":
        return GroqBackend() if os.environ.get("GROQ_API_KEY") else ScriptedBackend()
    raise ValueError(f"Unknown backend '{name}'. Use 'auto', 'groq' or 'scripted'.")
