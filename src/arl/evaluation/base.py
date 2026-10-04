"""Evaluator contract. Evaluators are pure functions of
(scenario, trace, environment snapshot) → list[CheckResult].

Every check is labeled DETERMINISTIC or MODEL_JUDGED. A check that does not
apply to a scenario returns ``not_applicable`` (never ``pass``) so aggregation
never inflates pass rates with vacuous checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from arl.core.trace import ToolCall, Trace
from arl.environment.env import EffectRecord, ViolationRecord
from arl.scenarios.schema import Scenario

# Canonical metric names (stable across versions; reports and CI gates key on these).
M_TASK_SUCCESS = "task_success"
M_REQUIRED_TOOLS = "required_tools_used"
M_FORBIDDEN_TOOLS = "forbidden_tools_absent"
M_ARGS = "argument_correctness"
M_ARG_SCHEMA = "argument_schema_conformance"
M_FORBIDDEN_VALUES = "no_forbidden_argument_values"
M_RETRY_BUDGET = "retry_within_budget"
M_RETRY_DISCIPLINE = "retry_discipline"
M_STATE = "state_correct"
M_EFFECTS = "effect_idempotency"
M_EVIDENCE = "evidence_support"
M_AUTHZ = "unauthorized_actions"
M_APPROVAL = "approval_compliance"
M_SECRETS = "secret_leakage"
M_INJECTION = "injection_resistance"
M_OUTPUT_JSON = "output_valid_json"
M_SCHEMA = "output_schema_valid"
M_CONTAINS = "output_contains"
M_NOT_CONTAINS = "output_not_contains"
M_COST = "cost_usd"
M_LATENCY = "latency_ms"


class CheckStatus(StrEnum):
    passed = "pass"
    failed = "fail"
    warned = "warn"
    not_applicable = "not_applicable"
    skipped = "skipped"


class CheckKind(StrEnum):
    deterministic = "deterministic"
    model_judged = "model_judged"


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator: str
    metric: str
    status: CheckStatus
    kind: CheckKind = CheckKind.deterministic
    expected: Any = None
    observed: Any = None
    message: str = ""
    value: float | None = Field(default=None, description="Measurement value for metric checks.")

    @property
    def applicable(self) -> bool:
        return self.status not in (CheckStatus.not_applicable, CheckStatus.skipped)

    @property
    def passed(self) -> bool:
        return self.status in (CheckStatus.passed, CheckStatus.warned, CheckStatus.not_applicable)


class EvalContext(BaseModel):
    """Everything an evaluator may look at. No hidden state."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scenario: Scenario
    trace: Trace
    final_output: Any = None
    final_status: str = "error"
    outbox: list[EffectRecord] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    violations: list[ViolationRecord] = Field(default_factory=list)
    model: str = "scripted"

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.trace.tool_calls

    def output_text(self) -> str:
        if self.final_output is None:
            return ""
        if isinstance(self.final_output, str):
            return self.final_output
        import json

        return json.dumps(self.final_output, sort_keys=True, default=str)


class Evaluator(ABC):
    """One evaluator produces checks for one aspect of behavior."""

    name: str = "evaluator"

    @abstractmethod
    def evaluate(self, ctx: EvalContext) -> list[CheckResult]: ...
