"""Structured outputs the model is required to produce.

Every LLM call in this package is bound to one of these schemas. Nothing parses
free text: the model either returns an object that validates, or the call is
retried. That is what makes `structured_output_validity` in the evaluation a
meaningful number rather than a measure of how good the regex was.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TriageCategory(str, Enum):
    """What kind of problem an anomaly most likely represents.

    Deliberately about *cause*, not severity - the detector already assigns
    severity from how far past threshold a metric sits. Category is the thing a
    threshold cannot tell you.
    """

    INFRASTRUCTURE = "INFRASTRUCTURE"   # venue/network/system slowness
    CLIENT_BEHAVIOUR = "CLIENT_BEHAVIOUR"  # one account acting unusually
    MARKET_ACTIVITY = "MARKET_ACTIVITY"    # legitimate large or unusual flow
    DATA_QUALITY = "DATA_QUALITY"          # the record itself looks wrong


class Classification(BaseModel):
    """Output of the classifier node."""

    category: TriageCategory
    urgency: int = Field(ge=1, le=5, description="1 = routine, 5 = page someone now.")
    rationale: str = Field(min_length=10, description="Why this category, in one sentence.")


class ToolChoice(BaseModel):
    """Output of the investigator node: which evidence to gather next."""

    tool_name: str = Field(description="Exact name of one available tool.")
    argument: str = Field(description="The single argument to pass to that tool.")
    why: str = Field(min_length=10, description="What this call is expected to establish.")


class Verdict(str, Enum):
    """Terminal routing decision."""

    AUTO_CLOSE = "AUTO_CLOSE"       # explained by evidence; no human needed
    ESCALATE = "ESCALATE"           # a human should look at this
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"  # loop, if budget remains


class Decision(BaseModel):
    """Output of the decision node."""

    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=15, description="The justification a human will read.")


class TriageRecord(BaseModel):
    """The complete audit trail for one anomaly.

    This is the deliverable. An agent that reaches a good decision without
    leaving a record of how is not usable in an operations context, so every
    intermediate step is retained rather than discarded once consumed.
    """

    anomaly_type: str
    entity_id: str
    severity: str
    classification: Classification | None = None
    tool_calls: list[dict] = Field(default_factory=list)
    decision: Decision | None = None
    iterations: int = 0
    error: str | None = None
