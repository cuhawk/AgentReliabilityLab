"""Task completion and output-assertion evaluators."""

from __future__ import annotations

from typing import Any

from arl.core.trace import Trace
from arl.evaluation.base import (
    M_CONTAINS,
    M_NOT_CONTAINS,
    M_OUTPUT_JSON,
    M_SCHEMA,
    M_TASK_SUCCESS,
    CheckResult,
    CheckStatus,
    EvalContext,
    Evaluator,
)


def final_status_of(trace: Trace) -> str:
    """The honest terminal status: platform-level outcomes (timeout,
    interrupted, cancelled) win; otherwise the agent's own AGENT_COMPLETE
    status; absence of both is an error."""
    if trace.status in ("timeout", "cancelled", "interrupted"):
        return trace.status
    complete = trace.complete_payload
    if complete is not None:
        return str(complete.get("status", "success"))
    if trace.error_events:
        return "error"
    return "error"


class CompletionEvaluator(Evaluator):
    name = "completion"

    def evaluate(self, ctx: EvalContext) -> list[CheckResult]:
        expect = ctx.scenario.expect
        checks: list[CheckResult] = []
        actual = ctx.final_status

        if expect.final_status is not None:
            passed = actual == expect.final_status
            checks.append(
                CheckResult(
                    evaluator=self.name,
                    metric=M_TASK_SUCCESS,
                    status=CheckStatus.passed if passed else CheckStatus.failed,
                    expected=expect.final_status,
                    observed=actual,
                    message="" if passed else f"agent ended '{actual}', expected '{expect.final_status}'",
                )
            )

        if expect.output_contains:
            text = ctx.output_text()
            missing = [s for s in expect.output_contains if s not in text]
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_CONTAINS,
                    status=CheckStatus.failed if missing else CheckStatus.passed,
                    expected=expect.output_contains, observed=missing or expect.output_contains,
                    message=f"missing substrings: {missing}" if missing else "",
                )
            )

        if expect.output_not_contains:
            text = ctx.output_text()
            present = [s for s in expect.output_not_contains if s in text]
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_NOT_CONTAINS,
                    status=CheckStatus.failed if present else CheckStatus.passed,
                    expected="none present", observed=present,
                    message=f"forbidden substrings present: {present}" if present else "",
                )
            )

        if expect.forbidden_event_types:
            present = sorted({e.type.value for e in ctx.trace.events} & set(expect.forbidden_event_types))
            checks.append(
                CheckResult(
                    evaluator=self.name, metric="no_forbidden_events",
                    status=CheckStatus.failed if present else CheckStatus.passed,
                    expected=f"none of {expect.forbidden_event_types}", observed=present,
                    message=f"forbidden event types occurred: {present}" if present else "",
                )
            )

        return checks


class OutputFormatEvaluator(Evaluator):
    """JSON validity and (lightweight) schema conformance of the final output."""

    name = "output_format"

    def evaluate(self, ctx: EvalContext) -> list[CheckResult]:
        expect = ctx.scenario.expect
        checks: list[CheckResult] = []

        if expect.output_json is not None:
            ok, err = _is_json(ctx.final_output)
            if not expect.output_json:
                checks.append(CheckResult(evaluator=self.name, metric=M_OUTPUT_JSON,
                                          status=CheckStatus.not_applicable))
            else:
                checks.append(CheckResult(evaluator=self.name, metric=M_OUTPUT_JSON,
                                          status=CheckStatus.passed if ok else CheckStatus.failed,
                                          expected="parseable JSON", observed=err or "ok",
                                          message=f"output is not valid JSON: {err}" if not ok else ""))

        if expect.output_schema is not None:
            errors = validate_against_schema(ctx.final_output, expect.output_schema)
            checks.append(
                CheckResult(
                    evaluator=self.name, metric=M_SCHEMA,
                    status=CheckStatus.failed if errors else CheckStatus.passed,
                    expected=expect.output_schema.get("required"),
                    observed=errors or "ok",
                    message=f"schema violations: {errors}" if errors else "",
                )
            )
        return checks


def _is_json(output: Any) -> tuple[bool, str | None]:
    import json

    if isinstance(output, str):
        try:
            json.loads(output)
            return True, None
        except (ValueError, TypeError) as exc:
            return False, str(exc)
    if isinstance(output, (dict, list, int, float, bool)) or output is None:
        return True, None  # structured value from the event payload
    return False, f"unserializable type {type(output).__name__}"


def validate_against_schema(value: Any, schema: dict[str, Any]) -> list[str]:
    """Minimal JSON-schema subset validator: type, required, properties, items,
    enum. Deliberately small — full JSON Schema is out of scope for v0 and the
    subset is documented in docs/SCENARIO_DSL.md."""
    errors: list[str] = []
    _validate_node(value, schema, "$", errors)
    return errors


_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def _validate_node(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected_type = schema.get("type")
    if expected_type:
        py_types = _TYPE_MAP.get(expected_type)
        if py_types and not isinstance(value, py_types):
            # bool is an int subclass; keep them distinct for "number"/"integer".
            if expected_type in ("number", "integer") and isinstance(value, bool):
                errors.append(f"{path}: expected {expected_type}, got boolean")
                return
            errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
        return
    if isinstance(value, dict) and (expected_type == "object" or "properties" in schema or "required" in schema):
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required property '{req}'")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                _validate_node(value[key], sub, f"{path}.{key}", errors)
    if isinstance(value, list) and expected_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                _validate_node(item, items, f"{path}[{i}]", errors)
