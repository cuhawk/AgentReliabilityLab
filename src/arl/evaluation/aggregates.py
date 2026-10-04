"""Suite-level metric aggregation. Every aggregate records its n and, for
rates, a Wilson 95% interval. Nothing here invents numbers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from arl.evaluation.base import (
    M_APPROVAL,
    M_ARG_SCHEMA,
    M_ARGS,
    M_AUTHZ,
    M_EVIDENCE,
    M_FORBIDDEN_VALUES,
    M_INJECTION,
    M_REQUIRED_TOOLS,
    M_SECRETS,
    M_TASK_SUCCESS,
)
from arl.evaluation.engine import ScenarioResult, metric_pass
from arl.evaluation.metrics import mean, percentile, wilson_interval


class MetricValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    unit: str = "ratio"
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None
    source: str = "deterministic"  # or "model_judged" when any input was judged

    def rounded(self, digits: int = 4) -> float:
        return round(self.value, digits)


def _rate(results: list[ScenarioResult], metric: str) -> MetricValue | None:
    applicable = [(r, metric_pass(r, metric)) for r in results]
    applicable = [(r, ok) for r, ok in applicable if ok is not None]
    if not applicable:
        return None
    successes = sum(1 for _, ok in applicable if ok)
    total = len(applicable)
    ci = wilson_interval(successes, total)
    return MetricValue(
        name=metric + "_rate", value=successes / total,
        ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None, n=total,
    )


def aggregate_suite(results: list[ScenarioResult]) -> dict[str, MetricValue]:
    out: dict[str, MetricValue] = {}
    if not results:
        return out

    passed = sum(1 for r in results if r.status == "pass")
    out["suite_pass_rate"] = MetricValue(name="suite_pass_rate", value=passed / len(results),
                                         n=len(results))

    for key, metric in [
        ("task_success_rate", M_TASK_SUCCESS),
        ("tool_selection_accuracy", M_REQUIRED_TOOLS),
        ("approval_compliance_rate", M_APPROVAL),
        ("injection_resistance_rate", M_INJECTION),
    ]:
        m = _rate(results, metric)
        if m is not None:
            m.name = key
            out[key] = m

    # Tool argument accuracy: all argument metrics must hold.
    arg_ok: list[bool | None] = []
    for r in results:
        verdicts = [metric_pass(r, m) for m in (M_ARGS, M_ARG_SCHEMA, M_FORBIDDEN_VALUES)]
        verdicts = [v for v in verdicts if v is not None]
        arg_ok.append(all(verdicts) if verdicts else None)
    applicable = [ok for ok in arg_ok if ok is not None]
    if applicable:
        ci = wilson_interval(sum(applicable), len(applicable))
        out["tool_argument_accuracy"] = MetricValue(
            name="tool_argument_accuracy", value=sum(applicable) / len(applicable),
            ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None, n=len(applicable),
        )

    # Recovery: among scenarios with injected faults, share that passed.
    faulted = [r for r in results if r.faults]
    if faulted:
        recovered = sum(1 for r in faulted if r.status == "pass")
        ci = wilson_interval(recovered, len(faulted))
        out["recovery_success_rate"] = MetricValue(
            name="recovery_success_rate", value=recovered / len(faulted),
            ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None, n=len(faulted),
        )

    unauthorized = _rate(results, M_AUTHZ)
    if unauthorized is not None:
        out["unauthorized_action_rate"] = MetricValue(
            name="unauthorized_action_rate", value=1.0 - unauthorized.value,
            ci_low=None, ci_high=None, n=unauthorized.n,
        )
    unsupported = _rate(results, M_EVIDENCE)
    if unsupported is not None:
        out["unsupported_claim_rate"] = MetricValue(
            name="unsupported_claim_rate", value=1.0 - unsupported.value, n=unsupported.n,
        )
    secret_fail = _rate(results, M_SECRETS)
    if secret_fail is not None:
        out["secret_leakage_rate"] = MetricValue(
            name="secret_leakage_rate", value=1.0 - secret_fail.value, n=secret_fail.n,
        )

    avg = mean([float(r.tool_calls) for r in results])
    if avg is not None:
        out["mean_tool_calls"] = MetricValue(name="mean_tool_calls", value=avg, unit="count", n=len(results))
    avg = mean([float(r.retries) for r in results])
    if avg is not None:
        out["mean_retries"] = MetricValue(name="mean_retries", value=avg, unit="count", n=len(results))

    latencies = [float(v) for r in results if (v := r.metrics.get("latency_ms")) is not None]
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    if p50 is not None:
        out["p50_latency_ms"] = MetricValue(name="p50_latency_ms", value=p50, unit="ms", n=len(latencies))
    if p95 is not None:
        out["p95_latency_ms"] = MetricValue(name="p95_latency_ms", value=p95, unit="ms", n=len(latencies))

    for key, attr in [("mean_input_tokens", "tokens_in"), ("mean_output_tokens", "tokens_out")]:
        avg = mean([float(getattr(r, attr)) for r in results])
        if avg is not None:
            out[key] = MetricValue(name=key, value=avg, unit="tokens", n=len(results))

    costs = [float(v) for r in results if (v := r.metrics.get("cost_usd")) is not None]
    avg = mean(costs) if costs else None
    if avg is not None:
        out["mean_cost_usd"] = MetricValue(name="mean_cost_usd", value=avg, unit="usd", n=len(costs))
    successful_costs = [float(v) for r in results
                        if r.status == "pass" and (v := r.metrics.get("cost_usd")) is not None]
    avg = mean(successful_costs) if successful_costs else None
    if avg is not None:
        out["cost_per_successful_task_usd"] = MetricValue(
            name="cost_per_successful_task_usd", value=avg, unit="usd", n=len(successful_costs),
        )
    return out
