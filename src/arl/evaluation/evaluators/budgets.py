"""Budget evaluators: cost and latency. These always *measure* (the value is
recorded even when the scenario sets no budget)."""

from __future__ import annotations

from arl.cost import cost_for_trace
from arl.evaluation.base import M_COST, M_LATENCY, CheckResult, CheckStatus, EvalContext, Evaluator


class CostEvaluator(Evaluator):
    name = "cost"

    def evaluate(self, ctx: EvalContext) -> list[CheckResult]:
        max_cost = ctx.scenario.expect.max_cost_usd
        cost, source = cost_for_trace(ctx.trace, ctx.model)
        if max_cost is None:
            return [
                CheckResult(evaluator=self.name, metric=M_COST, status=CheckStatus.passed,
                            value=round(cost, 6), expected=None, observed=None,
                            message=f"measured only (source: {source}); no budget set")
            ]
        ok = cost <= max_cost
        return [
            CheckResult(
                evaluator=self.name, metric=M_COST,
                status=CheckStatus.passed if ok else CheckStatus.failed,
                value=round(cost, 6),
                expected=f"<= ${max_cost:.4f} ({source})",
                observed=f"${cost:.4f}",
                message=f"cost ${cost:.4f} exceeds budget ${max_cost:.4f}" if not ok else "",
            )
        ]


class LatencyEvaluator(Evaluator):
    name = "latency"

    def evaluate(self, ctx: EvalContext) -> list[CheckResult]:
        max_latency = ctx.scenario.expect.max_latency_ms
        duration = ctx.trace.duration_ms
        if max_latency is None:
            return [
                CheckResult(evaluator=self.name, metric=M_LATENCY, status=CheckStatus.passed,
                            value=round(duration, 3), expected=None, observed=None,
                            message="measured only; no budget set")
            ]
        ok = duration <= max_latency
        return [
            CheckResult(
                evaluator=self.name, metric=M_LATENCY,
                status=CheckStatus.passed if ok else CheckStatus.failed,
                value=round(duration, 3),
                expected=f"<= {max_latency:.0f} ms (real wall-clock)",
                observed=f"{duration:.0f} ms",
                message=f"latency {duration:.0f} ms exceeds budget {max_latency:.0f} ms" if not ok else "",
            )
        ]
