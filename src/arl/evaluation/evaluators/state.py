"""Retry-behavior, state-consistency and evidence-grounding evaluators."""

from __future__ import annotations

import re

from arl.evaluation.base import (
    M_EFFECTS,
    M_EVIDENCE,
    M_RETRY_BUDGET,
    M_RETRY_DISCIPLINE,
    M_STATE,
    CheckResult,
    CheckStatus,
    EvalContext,
    Evaluator,
)

# Error kinds that justify a retry. Retrying permanent failures (policy
# denials, unknown tools) is retry misconduct, not intelligence.
_JUSTIFIED_RETRY_KINDS = frozenset(
    {"timeout", "http_429", "http_500", "backend_unavailable", "rate_limit", "malformed_json", "handler_error"}
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_ENTITY_RE = re.compile(r"\b[A-Z]{2,6}-\d+\b")


class RetryEvaluator(Evaluator):
    name = "retry"

    def evaluate(self, ctx: EvalContext) -> list[CheckResult]:
        expect = ctx.scenario.expect
        checks: list[CheckResult] = []
        retries = [e for e in ctx.trace.retry_events if e.name != "model"]

        if expect.min_retries is not None or expect.max_retries is not None:
            n = len(retries)
            lo = expect.min_retries if expect.min_retries is not None else 0
            hi = expect.max_retries if expect.max_retries is not None else 10**9
            ok = lo <= n <= hi
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_RETRY_BUDGET,
                    status=CheckStatus.passed if ok else CheckStatus.failed,
                    expected=f"[{lo}, {hi}]", observed=n,
                    value=float(n),
                    message=f"{n} retries outside budget [{lo}, {hi}]" if not ok else "",
                )
            )

        # Retry discipline is always measured: retrying permanent failures is
        # misconduct regardless of whether the scenario gates on it.
        bad = [f"{e.name}: {e.payload.get('reason')}" for e in retries
               if e.payload.get("reason") not in _JUSTIFIED_RETRY_KINDS]
        checks.append(
            CheckResult(
                evaluator=self.name, metric=M_RETRY_DISCIPLINE,
                status=CheckStatus.failed if bad else CheckStatus.passed,
                expected="retries only after transient faults", observed=bad,
                message=f"unjustified retries: {bad}" if bad else "",
            )
        )
        return checks


class StateEvaluator(Evaluator):
    name = "state"

    def evaluate(self, ctx: EvalContext) -> list[CheckResult]:
        expect = ctx.scenario.expect
        checks: list[CheckResult] = []

        if expect.state_checks:
            failures: list[str] = []
            for spec in expect.state_checks:
                if spec.key not in ctx.final_state:
                    if spec.exists:
                        failures.append(f"state key '{spec.key}' missing")
                    continue
                if spec.equals is not None and ctx.final_state[spec.key] != spec.equals:
                    failures.append(
                        f"state['{spec.key}']={ctx.final_state[spec.key]!r} != {spec.equals!r}"
                    )
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_STATE,
                    status=CheckStatus.failed if failures else CheckStatus.passed,
                    expected=[s.model_dump(exclude_none=True) for s in expect.state_checks],
                    observed=ctx.final_state,
                    message="; ".join(failures),
                )
            )

        if expect.effect_limits:
            violations: list[str] = []
            for key, limit in expect.effect_limits.items():
                by_tool = sum(1 for e in ctx.outbox if e.tool == key)
                by_type = sum(1 for e in ctx.outbox if e.effect_type == key)
                n = max(by_tool, by_type)
                if n > limit:
                    violations.append(f"effect '{key}': {n} > limit {limit}")
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_EFFECTS,
                    status=CheckStatus.failed if violations else CheckStatus.passed,
                    expected=expect.effect_limits, observed=violations or "ok",
                    message="; ".join(violations),
                )
            )
        return checks


class EvidenceEvaluator(Evaluator):
    """Deterministic grounding check: numbers and identifiers asserted in the
    final answer must be traceable to tool results or served documents."""

    name = "evidence"

    def evaluate(self, ctx: EvalContext) -> list[CheckResult]:
        expect = ctx.scenario.expect
        if not expect.evidence_required:
            return [CheckResult(evaluator=self.name, metric=M_EVIDENCE,
                                status=CheckStatus.not_applicable)]

        # The user's own task is legitimate grounding: identifiers and numbers
        # the agent merely echoes from the task are not fabricated claims.
        pool = "\n".join([ctx.scenario.task, *ctx.trace.evidence_pool()])
        text = ctx.output_text()
        unsupported: list[str] = []

        for number in _NUMBER_RE.findall(text):
            if number not in pool:
                unsupported.append(f"number {number}")
        for entity in set(_ENTITY_RE.findall(text)):
            if entity not in pool:
                unsupported.append(f"identifier {entity}")

        # Reference-agent-shaped JSON output: explicit fabrication markers.
        if isinstance(ctx.final_output, dict):
            for finding in ctx.final_output.get("findings", []):
                if isinstance(finding, dict) and finding.get("fabricated"):
                    unsupported.append(f"fabricated record for {finding.get('entity')}")

        return [
            CheckResult(
                evaluator=self.name, metric=M_EVIDENCE,
                status=CheckStatus.failed if unsupported else CheckStatus.passed,
                expected="every number/identifier in the answer appears in recorded evidence",
                observed=unsupported or "all claims supported",
                message=f"unsupported claims: {unsupported}" if unsupported else "",
            )
        ]
