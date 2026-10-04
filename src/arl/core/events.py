"""Typed event model. Events are the universal currency of AgentReliabilityLab:

every observable thing an agent does (model calls, tool calls, retries, approvals,
policy decisions, injected faults, state changes) is recorded as an :class:`Event`
with a monotonic sequence number. Evaluators and metrics are pure functions of the
event stream plus the scenario — there is no hidden scoring state.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """Normalized lifecycle events emitted by every agent adapter.

    The first ten are the contract from the architecture brief; the remaining
    events record *outcomes* (decisions, injections, evidence) that evaluators
    would otherwise have to infer.
    """

    AGENT_START = "agent_start"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_REQUEST = "tool_request"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    RETRY = "retry"
    HUMAN_APPROVAL_REQUEST = "human_approval_request"
    HUMAN_APPROVAL_RESULT = "human_approval_result"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"

    # Outcomes / platform bookkeeping.
    POLICY_CHECK = "policy_check"
    FAULT_INJECTED = "fault_injected"
    EVIDENCE_UPDATE = "evidence_update"


class RiskClass(StrEnum):
    """Tool risk taxonomy used by the authorization policy engine."""

    READ = "read"
    WRITE = "write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    PRIVILEGED = "privileged"


class Event(BaseModel):
    """One recorded occurrence in an agent run."""

    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=0, description="Monotonic within the run, 0-based.")
    t_offset_ms: float = Field(ge=0, description="Milliseconds since run start (monotonic).")
    wall_time: float = Field(description="Unix timestamp of recording.")
    type: EventType
    run_id: str
    scenario_id: str
    agent_id: str = "unknown"
    name: str | None = Field(default=None, description="Tool name, model id or label.")
    tool_call_id: str | None = None
    risk_class: RiskClass | None = None
    attempt: int = Field(default=1, ge=1, description="1-based attempt for this logical action.")
    payload: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        """Short human-readable line used by trace rendering."""
        bits = [self.type.value.upper()]
        if self.name:
            bits.append(self.name)
        for key in ("fault", "status", "decision", "reason", "tool", "attempt"):
            if key in self.payload:
                bits.append(f"{key}={self.payload[key]}")
        return " ".join(str(b) for b in bits)


def event_kind_of(trace_event: Event) -> str:
    """Bucket an event for span/metric purposes."""
    if trace_event.type is EventType.TOOL_REQUEST:
        return "tool"
    if trace_event.type in (EventType.MODEL_REQUEST, EventType.MODEL_RESPONSE):
        return "model"
    if trace_event.type is EventType.RETRY:
        return "retry"
    return trace_event.type.value
