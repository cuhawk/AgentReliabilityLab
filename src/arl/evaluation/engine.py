"""Evaluation engine: run all evaluators over a finished run, produce a
ScenarioResult, and aggregate suite-level metrics with confidence intervals."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from arl.core.trace import Trace
from arl.environment.env import EffectRecord, ViolationRecord
from arl.evaluation.base import (
    CheckResult,
    CheckStatus,
    EvalContext,
    Evaluator,
)
from arl.evaluation.evaluators.budgets import CostEvaluator, LatencyEvaluator
from arl.evaluation.evaluators.completion import CompletionEvaluator, OutputFormatEvaluator, final_status_of
from arl.evaluation.evaluators.security import (
    ApprovalEvaluator,
    AuthorizationEvaluator,
    InjectionEvaluator,
    SecretsEvaluator,
)
from arl.evaluation.evaluators.state import EvidenceEvaluator, RetryEvaluator, StateEvaluator
from arl.evaluation.evaluators.tools import ArgsEvaluator, ToolsEvaluator
from arl.scenarios.schema import Scenario

ALL_EVALUATORS: list[Evaluator] = [
    CompletionEvaluator(),
    OutputFormatEvaluator(),
    ToolsEvaluator(),
    ArgsEvaluator(),
    RetryEvaluator(),
    StateEvaluator(),
    EvidenceEvaluator(),
    AuthorizationEvaluator(),
    ApprovalEvaluator(),
    SecretsEvaluator(),
    InjectionEvaluator(),
    CostEvaluator(),
    LatencyEvaluator(),
]


class ScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scenario_id: str
    title: str = ""
    category: str = ""
    agent_id: str = "unknown"
    model: str = "scripted"
    status: str = "pass"  # pass | fail
    trace_status: str = "completed"
    final_status: str = "error"
    duration_ms: float = 0.0
    checks: list[CheckResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    faults: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    violations: list[ViolationRecord] = Field(default_factory=list)
    trace_text: str = ""
    trace: Trace | None = Field(default=None, exclude=True)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.failed]

    def check(self, metric: str) -> CheckResult | None:
        return next((c for c in self.checks if c.metric == metric), None)


def evaluate_scenario(
    scenario: Scenario,
    trace: Trace,
    *,
    outbox: list[EffectRecord],
    final_state: dict[str, Any],
    secrets: dict[str, str],
    violations: list[ViolationRecord] | None = None,
    model: str = "scripted",
) -> ScenarioResult:
    ctx = EvalContext(
        scenario=scenario,
        trace=trace,
        final_output=trace.final_output,
        final_status=final_status_of(trace),
        outbox=outbox,
        final_state=final_state,
        secrets=secrets,
        violations=violations or [],
        model=model,
    )
    checks: list[CheckResult] = []
    for evaluator in ALL_EVALUATORS:
        try:
            checks.extend(evaluator.evaluate(ctx))
        except Exception as exc:
            checks.append(
                CheckResult(
                    evaluator=evaluator.name, metric=f"{evaluator.name}_crashed",
                    status=CheckStatus.failed, observed=f"{type(exc).__name__}: {exc}",
                    message="evaluator crashed — treat the scenario as failed",
                )
            )
    tokens_in, tokens_out = trace.token_usage()
    retries = len(trace.retry_events)
    result = ScenarioResult(
        scenario_id=scenario.id,
        title=scenario.title_or_id,
        category=scenario.category.value,
        agent_id=trace.agent_id,
        model=model,
        status="fail" if any(c.status is CheckStatus.failed for c in checks) else "pass",
        trace_status=trace.status,
        final_status=ctx.final_status,
        duration_ms=trace.duration_ms,
        checks=checks,
        metrics={
            "latency_ms": round(trace.duration_ms, 3),
            "cost_usd": round(cost_of(checks), 6),
            "tool_calls": float(len(trace.tool_calls)),
            "retries": float(retries),
            "tokens_in": float(tokens_in),
            "tokens_out": float(tokens_out),
        },
        faults=trace.effective_faults(),
        tool_calls=len(trace.tool_calls),
        retries=retries,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        violations=list(ctx.violations),
        trace_text=trace.render_text(),
        trace=trace,
    )
    return result


def cost_of(checks: list[CheckResult]) -> float:
    for c in checks:
        if c.metric == "cost_usd" and c.value is not None:
            return c.value
    return 0.0


def metric_pass(result: ScenarioResult, metric: str) -> bool | None:
    """True/False when the metric was applicable for this scenario, else None."""
    check = result.check(metric)
    if check is None or not check.applicable:
        return None
    return check.status in (CheckStatus.passed, CheckStatus.warned)
