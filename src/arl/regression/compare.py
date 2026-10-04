"""Baseline vs candidate comparison — the CI gate.

Rules (documented in EVALUATION.md):
  * A gate fires only when its configured threshold is violated.
  * Deltas below the threshold's ``min_effect`` noise floor never fire a gate.
  * When baseline and candidate Wilson intervals for a rate metric do not
    overlap, the row is flagged ``statistically_significant`` — informational.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from arl.evaluation.aggregates import MetricValue
from arl.evaluation.metrics import intervals_overlap
from arl.regression.thresholds import MetricThreshold


class ComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = None
    delta_pct: float | None = None
    baseline_ci: tuple[float, float] | None = None
    candidate_ci: tuple[float, float] | None = None
    statistically_significant: bool | None = None
    threshold: str = ""
    status: str = "not_comparable"  # pass | regression | pass_low_effect | not_comparable
    note: str = ""


class ScenarioRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    baseline: str | None = None
    candidate: str | None = None


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_id: str = Field(default_factory=lambda: f"cmp-{uuid.uuid4().hex[:12]}")
    created_at: float = Field(default_factory=time.time)
    baseline_run_id: str = ""
    candidate_run_id: str = ""
    verdict: str = "pass"  # pass | regression
    rows: list[ComparisonRow] = Field(default_factory=list)
    scenario_rows: list[ScenarioRow] = Field(default_factory=list)
    thresholds: list[MetricThreshold] = Field(default_factory=list)

    @property
    def regressions(self) -> list[str]:
        return [r.metric for r in self.rows if r.status == "regression"]

    def render_table(self) -> str:
        header = f"{'Metric':34} {'Baseline':>12} {'Candidate':>12} {'Delta':>12}  Gate"
        lines = [header, "-" * len(header)]
        for row in self.rows:
            base = f"{row.baseline:.4f}" if row.baseline is not None else "n/a"
            cand = f"{row.candidate:.4f}" if row.candidate is not None else "n/a"
            delta = ""
            if row.delta is not None:
                pct = f" ({row.delta_pct:+.1f}%)" if row.delta_pct is not None else ""
                delta = f"{row.delta:+.4f}{pct}"
            lines.append(f"{row.metric:34} {base:>12} {cand:>12} {delta:>12}  {row.status}")
        return "\n".join(lines)


def compare_metrics(
    baseline: dict[str, MetricValue],
    candidate: dict[str, MetricValue],
    thresholds: list[MetricThreshold],
    *,
    baseline_run_id: str = "",
    candidate_run_id: str = "",
) -> Comparison:
    comparison = Comparison(baseline_run_id=baseline_run_id, candidate_run_id=candidate_run_id,
                            thresholds=thresholds)
    regressions = 0
    for threshold in thresholds:
        base = baseline.get(threshold.metric)
        cand = candidate.get(threshold.metric)
        row = ComparisonRow(metric=threshold.metric, threshold=threshold.describe())
        if base is not None:
            row.baseline = base.value
            row.baseline_ci = _ci_pair(base)
        if cand is not None:
            row.candidate = cand.value
            row.candidate_ci = _ci_pair(cand)
        if row.baseline is None or row.candidate is None:
            row.note = "metric missing on one side"
            comparison.rows.append(row)
            continue

        row.delta = row.candidate - row.baseline
        if row.baseline not in (0.0,) and abs(row.baseline) > 0:
            row.delta_pct = row.delta / abs(row.baseline) * 100.0
        if row.baseline_ci and row.candidate_ci:
            row.statistically_significant = not intervals_overlap(row.baseline_ci, row.candidate_ci)

        status = _gate(threshold, row)
        row.status = status
        if status == "regression":
            regressions += 1
        comparison.rows.append(row)

    for metric in sorted(set(baseline) | set(candidate)):
        if metric not in {t.metric for t in thresholds}:
            row = ComparisonRow(metric=metric)
            if metric in baseline and metric in candidate:
                row.baseline = baseline[metric].value
                row.candidate = candidate[metric].value
                row.delta = row.candidate - row.baseline
                row.status = "pass"
                row.note = "informational (no threshold)"
            else:
                row.note = "metric missing on one side"
            comparison.rows.append(row)

    comparison.verdict = "regression" if regressions else "pass"
    return comparison


def _ci_pair(m: MetricValue) -> tuple[float, float] | None:
    """CI tuple only when both bounds are present (CI Low and High move together)."""
    if m.ci_low is not None and m.ci_high is not None:
        return (m.ci_low, m.ci_high)
    return None


def _gate(threshold: MetricThreshold, row: ComparisonRow) -> str:
    """Decide one row. Semantics:
      * ``candidate_max`` is a hard cap on the candidate value (no noise floor).
      * Drops only violate when a drop gate is configured AND the absolute
        delta exceeds ``min_effect``; increases analogously.
      * A threshold only gates the direction(s) it declares — an increase on a
        metric with only drop gates is an improvement."""
    assert row.delta is not None

    if (threshold.candidate_max is not None and row.candidate is not None
            and row.candidate > threshold.candidate_max):
        row.note = f"candidate {row.candidate:.4f} exceeds hard cap {threshold.candidate_max}"
        return "regression"

    delta = row.delta
    if abs(delta) <= threshold.min_effect:
        return "pass_low_effect"

    if delta < 0:
        violates = False
        if threshold.max_drop is not None and abs(delta) > threshold.max_drop:
            violates = True
        if (
            threshold.max_drop_pct is not None
            and row.delta_pct is not None
            and abs(row.delta_pct) > threshold.max_drop_pct
        ):
            violates = True
        if violates:
            row.note = f"dropped {delta:+.4f} ({row.delta_pct:+.1f}%) — limit: {threshold.describe()}"
            return "regression"
        return "pass"

    violates = False
    if threshold.max_increase is not None and delta > threshold.max_increase:
        violates = True
    if (
        threshold.max_increase_pct is not None
        and row.delta_pct is not None
        and row.delta_pct > threshold.max_increase_pct
    ):
        violates = True
    if violates:
        row.note = f"increased {delta:+.4f} ({row.delta_pct:+.1f}%) — limit: {threshold.describe()}"
        return "regression"
    return "pass"


def compare_scenarios(
    baseline_results: list[dict[str, Any]], candidate_results: list[dict[str, Any]]
) -> list[ScenarioRow]:
    base = {sid: r.get("status") for r in baseline_results if (sid := r.get("scenario_id")) is not None}
    cand = {sid: r.get("status") for r in candidate_results if (sid := r.get("scenario_id")) is not None}
    return [
        ScenarioRow(scenario_id=sid, baseline=base.get(sid), candidate=cand.get(sid))
        for sid in sorted(set(base) | set(cand))
    ]
