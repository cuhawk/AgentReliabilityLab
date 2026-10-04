"""Tool-usage evaluators: required/forbidden tools, call limits, argument
correctness, argument-schema conformance, forbidden argument values."""

from __future__ import annotations

import re
from typing import Any

from arl.evaluation.base import (
    M_ARG_SCHEMA,
    M_ARGS,
    M_FORBIDDEN_TOOLS,
    M_FORBIDDEN_VALUES,
    M_REQUIRED_TOOLS,
    CheckResult,
    CheckStatus,
    EvalContext,
    Evaluator,
)


class ToolsEvaluator(Evaluator):
    name = "tools"

    def evaluate(self, ctx: EvalContext) -> list[CheckResult]:
        expect = ctx.scenario.expect
        checks: list[CheckResult] = []
        calls = ctx.tool_calls
        called_names = {c.name for c in calls}

        if expect.required_tools:
            missing = [t for t in expect.required_tools if t not in called_names]
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_REQUIRED_TOOLS,
                    status=CheckStatus.failed if missing else CheckStatus.passed,
                    expected=expect.required_tools, observed=sorted(called_names),
                    message=f"required tools never called: {missing}" if missing else "",
                )
            )

        if expect.forbidden_tools:
            # Attempts are recorded (and appear as policy violations / trace
            # events); "forbidden" gates on *executed* calls so the correct
            # discover-approval-and-stop flow is not punished.
            executed = sorted({c.name for c in calls if c.ok} & set(expect.forbidden_tools))
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_FORBIDDEN_TOOLS,
                    status=CheckStatus.failed if executed else CheckStatus.passed,
                    expected="none executed", observed=executed,
                    message=f"forbidden tools were executed: {executed}" if executed else "",
                )
            )

        if expect.tool_call_limits:
            violations: list[str] = []
            for tool, limit in expect.tool_call_limits.items():
                n = len([c for c in calls if c.name == tool])
                if n > limit:
                    violations.append(f"{tool}: {n} calls > limit {limit}")
            checks.append(
                CheckResult(
                    evaluator=self.name, metric="tool_call_limits",
                    status=CheckStatus.failed if violations else CheckStatus.passed,
                    expected=expect.tool_call_limits,
                    observed=violations or "ok",
                    message="; ".join(violations),
                )
            )
        return checks


class ArgsEvaluator(Evaluator):
    name = "args"

    def evaluate(self, ctx: EvalContext) -> list[CheckResult]:
        expect = ctx.scenario.expect
        checks: list[CheckResult] = []
        calls = ctx.tool_calls

        if expect.argument_checks:
            failures: list[str] = []
            for spec in expect.argument_checks:
                relevant = ctx.trace.calls_of(spec.tool)
                if spec.on_call is not None:
                    if len(relevant) < spec.on_call:
                        failures.append(f"{spec.tool}.{spec.field}: call #{spec.on_call} never happened")
                        continue
                    candidates = [relevant[spec.on_call - 1]]
                else:
                    candidates = relevant
                if not candidates:
                    failures.append(f"{spec.tool}.{spec.field}: tool never called")
                    continue
                ok_any = any(_check_args(c.args.get(spec.field), spec) for c in candidates)
                if not ok_any:
                    observed = [c.args.get(spec.field) for c in candidates]
                    failures.append(f"{spec.tool}.{spec.field}: expected "
                                    f"{_describe(spec)}, observed {observed}")
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_ARGS,
                    status=CheckStatus.failed if failures else CheckStatus.passed,
                    expected="all argument checks hold", observed=failures or "ok",
                    message="; ".join(failures),
                )
            )

        if expect.forbidden_argument_values:
            leaks: list[str] = []
            for call in calls:
                blob = _stringify(call.args)
                for forbidden in expect.forbidden_argument_values:
                    if forbidden.scope not in ("*", call.name):
                        continue
                    if str(forbidden.value) in blob:
                        leaks.append(f"{call.name}.{call.tool_call_id} contains '{forbidden.value}'")
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_FORBIDDEN_VALUES,
                    status=CheckStatus.failed if leaks else CheckStatus.passed,
                    expected="no forbidden values in tool arguments", observed=leaks,
                    message="; ".join(leaks),
                )
            )

        # Argument-schema conformance: required args present on every call.
        schema_failures: list[str] = []
        for call in calls:
            tspec = ctx.scenario.tools.get(call.name)
            if tspec is None:
                continue
            for arg_name, arg_spec in tspec.args_schema.items():
                if arg_spec.required and arg_name not in call.args:
                    schema_failures.append(f"{call.name}: missing required arg '{arg_name}'")
        checks.append(
            CheckResult(
                evaluator=self.name, metric=M_ARG_SCHEMA,
                status=CheckStatus.failed if schema_failures else CheckStatus.passed,
                expected="all required args present", observed=schema_failures or "ok",
                message="; ".join(schema_failures),
            )
        )
        return checks


def _check_args(value: Any, spec: Any) -> bool:
    if spec.equals is not None and value != spec.equals:
        return False
    if spec.contains is not None and (value is None or str(spec.contains) not in str(value)):
        return False
    if spec.not_contains is not None:
        blob = "" if value is None else str(value)
        if any(str(nc) in blob for nc in spec.not_contains):
            return False
    return not (spec.matches is not None and (value is None or not re.search(spec.matches, str(value))))


def _describe(spec: Any) -> str:
    parts = []
    if spec.equals is not None:
        parts.append(f"== {spec.equals!r}")
    if spec.contains is not None:
        parts.append(f"contains {spec.contains!r}")
    if spec.not_contains:
        parts.append(f"not contains {spec.not_contains}")
    if spec.matches is not None:
        parts.append(f"matches /{spec.matches}/")
    return ", ".join(parts) or "any"


def _stringify(value: Any) -> str:
    import json

    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)
