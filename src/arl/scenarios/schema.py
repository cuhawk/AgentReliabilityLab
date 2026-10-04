"""Scenario DSL: declarative YAML/JSON definitions of a controlled agent test.

A scenario binds together the task, the mock tool environment (with fault
schedules), authorization policy, simulated human decisions, and the
expectations that deterministic evaluators will check.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arl.core.events import RiskClass

# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------


class FaultType(StrEnum):
    # Tool faults
    timeout = "timeout"
    http_429 = "http_429"
    http_500 = "http_500"
    malformed_json = "malformed_json"
    empty_result = "empty_result"
    delay = "delay"
    partial_result = "partial_result"
    permission_denied = "permission_denied"
    # Model faults
    invalid_structured_output = "invalid_structured_output"
    truncated_response = "truncated_response"
    backend_unavailable = "backend_unavailable"
    model_rate_limit = "model_rate_limit"
    # Retrieval faults
    irrelevant_doc = "irrelevant_doc"
    stale_doc = "stale_doc"
    conflicting_evidence = "conflicting_evidence"
    missing_source = "missing_source"
    malicious_doc = "malicious_doc"
    # State faults
    missing_state = "missing_state"
    stale_state = "stale_state"
    duplicate_event = "duplicate_event"
    interrupted = "interrupted"


TRANSIENT_TOOL_FAULTS = frozenset(
    {
        FaultType.timeout,
        FaultType.http_429,
        FaultType.http_500,
        FaultType.backend_unavailable,
        FaultType.model_rate_limit,
    }
)

MODEL_FAULTS = frozenset(
    {
        FaultType.invalid_structured_output,
        FaultType.truncated_response,
        FaultType.backend_unavailable,
        FaultType.model_rate_limit,
    }
)

RETRIEVAL_FAULTS = frozenset(
    {
        FaultType.irrelevant_doc,
        FaultType.stale_doc,
        FaultType.conflicting_evidence,
        FaultType.missing_source,
        FaultType.malicious_doc,
    }
)

STATE_FAULTS = frozenset({FaultType.missing_state, FaultType.stale_state})

RUN_FAULTS = frozenset({FaultType.duplicate_event, FaultType.interrupted})

TOOL_FAULTS = frozenset(
    {
        FaultType.timeout,
        FaultType.http_429,
        FaultType.http_500,
        FaultType.malformed_json,
        FaultType.empty_result,
        FaultType.delay,
        FaultType.partial_result,
        FaultType.permission_denied,
        *RETRIEVAL_FAULTS,
    }
)


class FaultStep(BaseModel):
    """A fault applied to the Nth call of a target (tool or the model).

    ``call`` is 1-based. ``every`` applies the fault to every Nth call.
    At least one of the two must be set; when both are, ``call`` wins for that
    ordinal and ``every`` covers the rest.
    """

    model_config = ConfigDict(extra="forbid")

    call: int | None = Field(default=None, ge=1)
    every: int | None = Field(default=None, ge=1)
    fault: FaultType
    params: dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    @model_validator(mode="after")
    def _check_target(self) -> FaultStep:
        if self.call is None and self.every is None:
            raise ValueError("fault step needs 'call' or 'every'")
        return self

    def applies_to(self, ordinal: int) -> bool:
        if self.call is not None:
            return self.call == ordinal
        return (
            self.every is not None
            and self.every > 0
            and ordinal % self.every == 0
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

ToolKind = Literal["data", "compute", "retrieve", "effect", "http"]


class ArgSpec(BaseModel):
    """JSON-schema-flavored description of one tool argument."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "integer", "boolean", "object", "array"] = "string"
    description: str = ""
    required: bool = True
    enum: list[str] | None = None


class ScopeSpec(BaseModel):
    """Argument scope constraint enforced by the policy engine, e.g.
    ``{field: account_id, allowed: [C-101]}``."""

    model_config = ConfigDict(extra="forbid")

    field: str
    allowed: list[str] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """A mock tool in the sandboxed environment.

    Kinds:
      * ``data``     - keyed lookup over ``params.records`` (list of row dicts).
      * ``compute``  - pure function from the handler registry (``params.function``).
      * ``retrieve`` - document search over ``params.corpus``.
      * ``effect``   - records an outbound action in an in-process outbox
                       (email/webhook/transfer/delete); never touches a network.
      * ``http``     - scripted HTTP-style endpoint with a status/body sequence.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = ""
    kind: ToolKind = "data"
    risk_class: RiskClass = RiskClass.READ
    args_schema: dict[str, ArgSpec] = Field(default_factory=dict)
    scope: ScopeSpec | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    faults: list[FaultStep] = Field(default_factory=list)
    # For effect tools: logical destination, checked against env allowlist.
    target: str | None = None

    @model_validator(mode="after")
    def _check_params_for_kind(self) -> ToolSpec:
        if self.kind == "data" and "records" not in self.params:
            raise ValueError(f"tool '{self.name}' kind=data requires params.records")
        if self.kind == "retrieve" and "corpus" not in self.params:
            raise ValueError(f"tool '{self.name}' kind=retrieve requires params.corpus")
        if self.kind == "compute" and "function" not in self.params:
            raise ValueError(f"tool '{self.name}' kind=compute requires params.function")
        if self.kind == "effect" and "effect_type" not in self.params:
            raise ValueError(f"tool '{self.name}' kind=effect requires params.effect_type")
        if self.kind == "http" and "script" not in self.params:
            raise ValueError(f"tool '{self.name}' kind=http requires params.script")
        return self


# ---------------------------------------------------------------------------
# Environment, policy, human simulation
# ---------------------------------------------------------------------------


class EnvSpec(BaseModel):
    """Sandbox environment: non-secret variables, initial state, network allowlist."""

    model_config = ConfigDict(extra="forbid")

    variables: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    allowlisted_domains: list[str] = Field(default_factory=lambda: ["api.company.example"])
    # Named fake secrets; values are deterministic canaries synthesized by the
    # runner (never read from the real environment).
    secrets: list[str] = Field(default_factory=list)


class PolicyAction(StrEnum):
    allow = "allow"
    allow_in_scope = "allow_in_scope"
    require_approval = "require_approval"
    deny = "deny"


DEFAULT_POLICY: dict[RiskClass, PolicyAction] = {
    RiskClass.READ: PolicyAction.allow,
    RiskClass.WRITE: PolicyAction.allow_in_scope,
    RiskClass.EXTERNAL_SIDE_EFFECT: PolicyAction.require_approval,
    RiskClass.PRIVILEGED: PolicyAction.deny,
}


class PolicySpec(BaseModel):
    """Per-risk-class policy overrides merged over :data:`DEFAULT_POLICY`."""

    model_config = ConfigDict(extra="forbid")

    rules: dict[RiskClass, PolicyAction] = Field(default_factory=dict)

    def effective(self) -> dict[RiskClass, PolicyAction]:
        merged = dict(DEFAULT_POLICY)
        merged.update(self.rules)
        return merged


class HumanDecision(StrEnum):
    approve = "approve"
    deny = "deny"
    timeout = "timeout"


class HumanSpec(BaseModel):
    """How the simulated human responds to approval requests."""

    model_config = ConfigDict(extra="forbid")

    mode: HumanDecision = HumanDecision.approve
    delay_ms: float = Field(default=0.0, ge=0)
    timeout_ms: float = Field(default=500.0, ge=0)


# ---------------------------------------------------------------------------
# Expectations (evaluated deterministically)
# ---------------------------------------------------------------------------

FinalStatus = Literal["success", "failed", "blocked", "error", "interrupted", "timeout"]


class ArgumentCheck(BaseModel):
    """Assert something about the arguments an agent passed to a tool."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    field: str
    equals: Any | None = None
    contains: str | None = None
    not_contains: list[str] | None = None
    matches: str | None = None  # regex
    on_call: int | None = Field(default=None, ge=1, description="1-based call ordinal; None = all calls")


class ForbiddenValue(BaseModel):
    """A value that must never appear in tool arguments (exfiltration checks)."""

    model_config = ConfigDict(extra="forbid")

    value: str
    scope: str = "*"  # tool name or "*" for all tools


class StateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    equals: Any | None = None
    exists: bool = True


class Expectations(BaseModel):
    """What deterministic evaluators check. Absent fields are not evaluated."""

    model_config = ConfigDict(extra="forbid")

    # Task completion / output
    final_status: FinalStatus | None = None
    output_contains: list[str] = Field(default_factory=list)
    output_not_contains: list[str] = Field(default_factory=list)
    output_json: bool | None = None
    output_schema: dict[str, Any] | None = None

    # Tool behavior
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    tool_call_limits: dict[str, int] = Field(default_factory=dict)
    argument_checks: list[ArgumentCheck] = Field(default_factory=list)
    forbidden_argument_values: list[ForbiddenValue] = Field(default_factory=list)

    # Recovery
    min_retries: int | None = Field(default=None, ge=0)
    max_retries: int | None = Field(default=None, ge=0)
    retry_only_transient: bool = False  # RETRY events must follow transient faults

    # State
    state_checks: list[StateCheck] = Field(default_factory=list)
    # Max recorded outbound effects per tool (or effect type), e.g. {"send_email": 1}
    effect_limits: dict[str, int] = Field(default_factory=dict)

    # Grounding
    evidence_required: bool = False

    # Authorization / HITL
    approval_required_for: list[str] = Field(default_factory=list)
    no_unauthorized_actions: bool = False  # hard-gate: zero policy violations

    # Budgets
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_latency_ms: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)

    # Security
    injection_resistance: bool | None = None  # None = auto-enable when malicious content exists
    secrets_must_not_leak: bool = True
    # Event-stream shape: event types that must never occur (e.g. over-asking).
    forbidden_event_types: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenario root
# ---------------------------------------------------------------------------


class ScenarioCategory(StrEnum):
    happy_path = "happy_path"
    routing = "routing"
    tool_failure = "tool_failure"
    model_failure = "model_failure"
    retrieval_failure = "retrieval_failure"
    state_recovery = "state_recovery"
    authorization = "authorization"
    human_approval = "human_approval"
    prompt_injection = "prompt_injection"
    secret_handling = "secret_handling"
    hallucination = "hallucination"
    cost = "cost"
    latency = "latency"


_SCENARIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-_.]{2,80}$")


def _check_fault_domain(field: str, steps: list[FaultStep], allowed: frozenset[FaultType]) -> None:
    for step in steps:
        if step.fault not in allowed:
            legal = ", ".join(sorted(f.value for f in allowed))
            raise ValueError(f"{field}: fault '{step.fault.value}' not allowed here (legal: {legal})")


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    category: ScenarioCategory = ScenarioCategory.happy_path
    description: str = ""
    task: str
    tags: list[str] = Field(default_factory=list)
    tools: dict[str, ToolSpec] = Field(default_factory=dict)
    model_faults: list[FaultStep] = Field(default_factory=list)
    # Faults on the agent's private state store, keyed by state-read ordinal.
    state_faults: list[FaultStep] = Field(default_factory=list)
    # Whole-run faults keyed by global tool-call ordinal: duplicate_event
    # replays the previous tool result after call N; interrupted aborts the
    # run when call N starts.
    run_faults: list[FaultStep] = Field(default_factory=list)
    env: EnvSpec = Field(default_factory=EnvSpec)
    policy: PolicySpec = Field(default_factory=PolicySpec)
    human: HumanSpec = Field(default_factory=HumanSpec)
    timeout_ms: float = Field(default=30000, gt=0)
    max_steps: int = Field(default=12, ge=1)
    expect: Expectations = Field(default_factory=Expectations)
    source_file: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_cross_refs(self) -> Scenario:
        for key, spec in self.tools.items():
            if spec.name != key:
                raise ValueError(f"tool key '{key}' does not match tool name '{spec.name}'")
        for req in self.expect.required_tools:
            if req not in self.tools:
                raise ValueError(f"expect.required_tools references unknown tool '{req}'")
        for req in self.expect.forbidden_tools:
            if req not in self.tools:
                raise ValueError(f"expect.forbidden_tools references unknown tool '{req}'")
        for chk in self.expect.argument_checks:
            if chk.tool not in self.tools:
                raise ValueError(f"expect.argument_checks references unknown tool '{chk.tool}'")
        for tool in self.expect.approval_required_for:
            if tool not in self.tools:
                raise ValueError(f"expect.approval_required_for references unknown tool '{tool}'")
        if (
            self.expect.min_retries is not None
            and self.expect.max_retries is not None
            and self.expect.min_retries > self.expect.max_retries
        ):
            raise ValueError("expect.min_retries > expect.max_retries")
        _check_fault_domain("model_faults", self.model_faults, MODEL_FAULTS)
        _check_fault_domain("state_faults", self.state_faults, STATE_FAULTS)
        _check_fault_domain("run_faults", self.run_faults, RUN_FAULTS)
        for spec in self.tools.values():
            _check_fault_domain(f"tools.{spec.name}.faults", spec.faults, TOOL_FAULTS)
        return self

    def has_malicious_content(self) -> bool:
        """True when the scenario serves content flagged malicious - used to
        auto-enable the injection-resistance evaluator."""
        for spec in self.tools.values():
            if spec.kind == "retrieve" and any(
                doc.get("malicious") for doc in spec.params.get("corpus", [])
            ):
                return True
        return any(step.fault is FaultType.malicious_doc for step in self.tools_faults_iter())

    def tools_faults_iter(self) -> Iterator[FaultStep]:
        for spec in self.tools.values():
            yield from spec.faults

    @property
    def title_or_id(self) -> str:
        return self.title or self.id
