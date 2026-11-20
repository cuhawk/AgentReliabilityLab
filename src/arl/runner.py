"""Runner: orchestrates scenario execution — environment construction, agent
adapter, timeout handling, trace recording, evaluation, storage, reporting."""

from __future__ import annotations

import asyncio
import contextlib
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arl import __version__
from arl.adapters.registry import agent_id_of
from arl.environment.env import ToolEnvironment
from arl.environment.errors import EnvironmentInterrupted
from arl.evaluation.aggregates import MetricValue, aggregate_consistency, aggregate_suite
from arl.evaluation.engine import ScenarioResult, evaluate_scenario
from arl.faults.injector import FaultInjector
from arl.observability.tracing import EventSpanTracker, tracer
from arl.recording.recorder import TraceRecorder
from arl.recording.storage import SqliteStorage
from arl.scenarios.schema import Scenario
from arl.scheduler import EvaluationScheduler


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    agent_spec: str = "reference:guarded"
    agent_config: dict[str, Any] = Field(default_factory=dict)
    output_dir: Path = Path(".arl/runs")
    db_path: Path | None = None
    report_formats: list[str] = Field(default_factory=lambda: ["json", "md", "html"])
    trace_exporter: str = "none"  # none | console | otlp
    model: str | None = None  # cost-model override; inferred from agent spec otherwise
    trials: int = Field(default=1, ge=1)
    ks: list[int] = Field(default_factory=list)
    concurrency: int = Field(default=1, ge=1)
    retry_budget: int = Field(default=0, ge=0)
    seed: int = 0

    @model_validator(mode="after")
    def _check_ks(self) -> RunConfig:
        if any(k < 1 or k > self.trials for k in self.ks):
            raise ValueError("each k must be between 1 and the number of trials")
        return self


class SuiteRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    suite_id: str
    agent_id: str
    model: str
    created_at: float
    environment: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    results: list[ScenarioResult] = Field(default_factory=list)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    report_paths: dict[str, str] = Field(default_factory=dict)
    comparison: dict[str, Any] | None = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status != "pass")


def infer_model(agent_spec: str, override: str | None) -> str:
    if override:
        return override
    if agent_spec.startswith("openai:"):
        return agent_spec.split(":", 1)[1]
    return "scripted"


async def run_scenario(
    scenario: Scenario,
    *,
    run_id: str,
    agent_spec: str,
    cfg: RunConfig,
    scheduler: EvaluationScheduler | None = None,
    trial: int = 1,
) -> ScenarioResult:
    """Execute one scenario against the configured agent and evaluate it."""
    agent_id = agent_id_of(agent_spec)
    recorder = TraceRecorder(run_id, scenario.id, agent_id)
    injector = FaultInjector(scenario, recorder)
    env = ToolEnvironment(scenario, recorder, injector)

    from arl.adapters.base import TaskInput
    from arl.adapters.registry import create_adapter

    task_input = TaskInput(
        task=scenario.task,
        prior_turns=scenario.prior_turns,
        scenario_id=scenario.id,
        tools=env.manifest(),
        env={"variables": env.variables, "allowlisted_domains": scenario.env.allowlisted_domains},
        state=dict(scenario.env.state),
        max_steps=scenario.max_steps,
    )
    adapter = create_adapter(agent_spec, recorder=recorder, env=env, task_input=task_input,
                             agent_config=cfg.agent_config)
    adapter.scheduler = scheduler
    adapter.request_key = f"{scenario.id}:{trial}"
    recorder.agent_id = adapter.agent_id
    env.set_duplicate_sink(adapter.receive_event)

    span_tracker = EventSpanTracker()
    recorder.add_observer(span_tracker)

    status = "completed"
    # The agent.task span is owned by the EventSpanTracker (opened on
    # AGENT_START, closed on AGENT_COMPLETE/AGENT_ERROR) so that every tool,
    # model, policy and approval span nests beneath it.
    try:
        await asyncio.wait_for(adapter.start_task(), timeout=scenario.timeout_ms / 1000.0)
    except TimeoutError:
        status = "timeout"
        with contextlib.suppress(Exception):
            await adapter.cancel()
    except EnvironmentInterrupted:
        status = "interrupted"
    except Exception as exc:
        status = "error"
        await recorder.emit_error(f"{type(exc).__name__}: {exc}", kind="adapter_error")
    finally:
        await adapter.aclose()

    trace = await recorder.finish(status)
    span_tracker.close_abandoned()

    with tracer().start_as_current_span("evaluators", attributes={"arl.scenario": scenario.id}):
        result = evaluate_scenario(
            scenario, trace,
            outbox=env.outbox,
            final_state=env.final_state,
            secrets=env.secrets,
            violations=list(env.violations),
            model=infer_model(agent_spec, cfg.model),
        )
    result.trial = trial
    return result


async def run_suite(
    suite_id: str,
    scenarios: list[Scenario],
    cfg: RunConfig,
    *,
    storage: SqliteStorage | None = None,
    baseline_report: dict[str, Any] | None = None,
    thresholds: list | None = None,
) -> SuiteRun:
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    agent_id = agent_id_of(cfg.agent_spec)
    model = infer_model(cfg.agent_spec, cfg.model)

    run = SuiteRun(
        run_id=run_id,
        suite_id=suite_id,
        agent_id=agent_id,
        model=model,
        created_at=time.time(),
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "arl_version": __version__,
        },
        config=cfg.model_dump(
            include={"agent_spec", "trials", "ks", "concurrency", "retry_budget", "seed"},
        ) | ({"agent_config": cfg.agent_config} if cfg.agent_config else {}),
    )

    scheduler = EvaluationScheduler(cfg.concurrency, cfg.retry_budget, cfg.seed)

    async def execute(scenario: Scenario, trial: int) -> ScenarioResult:
        async with scheduler.slot():
            with tracer().start_as_current_span(
                f"eval.scenario {scenario.id}",
                attributes={"arl.scenario": scenario.id, "arl.trial": trial},
            ):
                return await run_scenario(
                    scenario, run_id=run_id, agent_spec=cfg.agent_spec, cfg=cfg, scheduler=scheduler, trial=trial,
                )

    with tracer().start_as_current_span(
        f"eval.run {suite_id}",
        attributes={"arl.suite": suite_id, "arl.agent": agent_id, "arl.scenarios": len(scenarios)},
    ):
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(execute(scenario, trial))
                for scenario in scenarios
                for trial in range(1, cfg.trials + 1)
            ]
        run.results = [task.result() for task in tasks]

    ks = cfg.ks or sorted({1, cfg.trials})
    run.metrics = aggregate_suite(run.results) | aggregate_consistency(run.results, ks)

    comparison_dict: dict[str, Any] | None = None
    if baseline_report is not None:
        from arl.evaluation.aggregates import MetricValue
        from arl.regression.compare import compare_metrics, compare_scenarios
        from arl.regression.thresholds import DEFAULT_THRESHOLDS

        effective_thresholds = thresholds if thresholds is not None else list(DEFAULT_THRESHOLDS)
        base_metrics = {
            name: MetricValue.model_validate(m) for name, m in baseline_report.get("metrics", {}).items()
        }
        comparison = compare_metrics(
            base_metrics, run.metrics, effective_thresholds,
            baseline_run_id=str(baseline_report.get("run_id", "")),
            candidate_run_id=run.run_id,
        )
        comparison.scenario_rows = compare_scenarios(
            baseline_report.get("scenarios", []),
            [{"scenario_id": r.scenario_id, "status": r.status} for r in run.results],
        )
        comparison_dict = comparison.model_dump()
        run.comparison = comparison_dict
        if storage is not None:
            storage.save_comparison(comparison.model_dump())

    if storage is not None:
        _persist(storage, run, cfg)

    from arl.reporting.render import write_reports

    run.report_paths = write_reports(run, cfg.output_dir, formats=cfg.report_formats,
                                     comparison=comparison_dict)
    if storage is not None:
        storage.save_run(_run_dict(run))
    return run


def _persist(storage: SqliteStorage, run: SuiteRun, cfg: RunConfig) -> None:
    from arl.evaluation.aggregates import MetricValue

    storage.save_run(_run_dict(run))
    for result in run.results:
        storage.save_scenario_result(run.run_id, result)
        if result.trace is not None:
            storage.save_events(run.run_id, result.result_id, result.trace.events)
            storage.save_tool_calls(run.run_id, result.result_id, result.trace.tool_calls)
        storage.save_violations(run.run_id, result.result_id, result.violations)
        scenario_metrics = {
            name: MetricValue(name=name, value=value, unit=_unit_for(name))
            for name, value in result.metrics.items()
        }
        storage.save_metrics(run.run_id, scenario_metrics, scenario_id=result.result_id)
    storage.save_metrics(run.run_id, run.metrics)


_UNIT_BY_METRIC = {"latency_ms": "ms", "cost_usd": "usd", "tool_calls": "count",
                   "retries": "count", "tokens_in": "tokens", "tokens_out": "tokens"}


def _unit_for(name: str) -> str:
    return _UNIT_BY_METRIC.get(name, "raw")


def _run_dict(run: SuiteRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "created_at": run.created_at,
        "suite": run.suite_id,
        "agent_id": run.agent_id,
        "model": run.model,
        "config": run.config,
        "environment": run.environment,
        "status": "completed",
        "scenario_count": len(run.results),
        "metrics": {k: v.model_dump() for k, v in run.metrics.items()},
        "report_paths": run.report_paths,
    }
