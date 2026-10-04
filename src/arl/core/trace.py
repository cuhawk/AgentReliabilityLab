"""Trace: the ordered event stream of one scenario run, plus derived views
(paired tool calls, retries, approvals, faults) that evaluators consume."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from arl.core.events import Event, EventType


class ToolCall(BaseModel):
    """One physical tool execution: a TOOL_REQUEST paired with its TOOL_RESULT."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    name: str
    attempt: int = 1
    args: dict[str, Any] = Field(default_factory=dict)
    request: Event
    result: Event | None = None

    @property
    def status(self) -> str:
        if self.result is None:
            return "pending"
        return str(self.result.payload.get("status", "unknown"))

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def error_kind(self) -> str | None:
        if self.result is None:
            return None
        return self.result.payload.get("error_kind")

    @property
    def fault(self) -> str | None:
        """Fault that was injected into this call, if any."""
        if self.result is None:
            return None
        return self.result.payload.get("fault")

    @property
    def result_value(self) -> Any:
        if self.result is None:
            return None
        return self.result.payload.get("result")

    @property
    def latency_ms(self) -> float | None:
        if self.result is None:
            return None
        return self.result.t_offset_ms - self.request.t_offset_ms


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    scenario_id: str
    agent_id: str = "unknown"
    status: str = "completed"  # completed | error | timeout | cancelled | interrupted
    started_at: float = 0.0
    duration_ms: float = 0.0
    events: list[Event] = Field(default_factory=list)

    # --- queries ---------------------------------------------------------

    def by_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.type is event_type]

    @property
    def final_output(self) -> Any:
        for e in self.by_type(EventType.AGENT_COMPLETE):
            return e.payload.get("output")
        return None

    @property
    def complete_payload(self) -> dict[str, Any] | None:
        for e in self.by_type(EventType.AGENT_COMPLETE):
            return e.payload
        return None

    @property
    def error_events(self) -> list[Event]:
        return self.by_type(EventType.AGENT_ERROR)

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls: dict[str, ToolCall] = {}
        for e in self.events:
            if e.type is EventType.TOOL_REQUEST:
                calls[e.tool_call_id or ""] = ToolCall(
                    tool_call_id=e.tool_call_id or "",
                    name=e.name or "unknown",
                    attempt=e.attempt,
                    args=dict(e.payload.get("args", {})),
                    request=e,
                )
            elif e.type is EventType.TOOL_RESULT and e.tool_call_id in calls:
                calls[e.tool_call_id].result = e
        return list(calls.values())

    def calls_of(self, tool_name: str) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.name == tool_name]

    @property
    def retry_events(self) -> list[Event]:
        return self.by_type(EventType.RETRY)

    @property
    def faults(self) -> list[Event]:
        return self.by_type(EventType.FAULT_INJECTED)

    @property
    def policy_checks(self) -> list[Event]:
        return self.by_type(EventType.POLICY_CHECK)

    def policy_denials(self) -> list[Event]:
        return [e for e in self.policy_checks if e.payload.get("decision") == "denied"]

    @property
    def approval_requests(self) -> list[Event]:
        return self.by_type(EventType.HUMAN_APPROVAL_REQUEST)

    def approval_results(self) -> list[Event]:
        return self.by_type(EventType.HUMAN_APPROVAL_RESULT)

    def approvals_granted(self) -> list[Event]:
        return [
            e for e in self.approval_results()
            if e.payload.get("decision") in ("approve", "approved")
        ]

    @property
    def evidence_updates(self) -> list[Event]:
        return self.by_type(EventType.EVIDENCE_UPDATE)

    @property
    def state_updates(self) -> list[Event]:
        return self.by_type(EventType.STATE_UPDATE)

    def model_calls(self) -> list[Event]:
        return self.by_type(EventType.MODEL_REQUEST)

    def token_usage(self) -> tuple[int, int]:
        """(input_tokens, output_tokens) summed from model events."""
        inp = out = 0
        for e in self.by_type(EventType.MODEL_RESPONSE):
            usage = e.payload.get("usage") or {}
            inp += int(usage.get("input_tokens", 0) or 0)
            out += int(usage.get("output_tokens", 0) or 0)
        return inp, out

    def evidence_pool(self) -> list[str]:
        """All content the agent could legitimately have based its answer on:
        tool results and served documents. Used by claim-support evaluation."""
        pool: list[str] = []
        for call in self.tool_calls:
            if call.ok and call.result_value is not None:
                pool.append(_stringify(call.result_value))
        for e in self.evidence_updates:
            for doc in e.payload.get("documents", []):
                pool.append(_stringify(doc))
        return pool

    def effective_faults(self) -> list[str]:
        """Faults that actually altered behavior (excluding pure delays)."""
        return sorted({f.payload.get("fault", "?") for f in self.faults})

    # --- rendering -------------------------------------------------------

    def render_text(self) -> str:
        """Human-readable timeline: ``00:03.412  TOOL_RESULT customer_api status=error``."""
        lines: list[str] = []
        minutes, seconds = divmod(self.duration_ms / 1000.0, 60)
        header = (
            f"trace scenario={self.scenario_id} agent={self.agent_id} "
            f"status={self.status} duration={int(minutes):02d}:{seconds:06.3f}"
        )
        lines.append(header)
        for e in self.events:
            t = e.t_offset_ms / 1000.0
            mm, ss = divmod(t, 60)
            lines.append(f"{int(mm):02d}:{ss:06.3f}  {e.summary()}")
        return "\n".join(lines)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)
