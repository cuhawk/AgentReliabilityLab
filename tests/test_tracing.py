"""Phase 8 acceptance: the event stream is rendered as nested OpenTelemetry
spans — eval.run → eval.scenario → agent.task → tool.x / llm.call / retry /
policy.check / approval — with fault, status and decision attributes intact.

These tests run real scenarios through the runner with an in-memory exporter;
the pipeline under test is not mocked."""

from __future__ import annotations

from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from arl.core.events import Event, EventType
from arl.observability.tracing import EventSpanTracker, setup_tracing
from arl.runner import RunConfig, run_suite
from arl.scenarios.loader import load_scenarios


def _one_scenario(scenario_dir: Path, scenario_id: str) -> list:
    found = load_scenarios([scenario_dir / f"{scenario_id}.yaml"])
    scenarios = found if isinstance(found, list) else [found]
    assert len(scenarios) == 1, f"expected exactly one scenario, got {len(scenarios)}"
    return scenarios


def _names(spans: list[ReadableSpan]) -> list[str]:
    return [s.name for s in spans]


def _first(spans: list[ReadableSpan], name: str) -> ReadableSpan:
    matches = [s for s in spans if s.name == name]
    assert matches, f"no span named {name!r}; got {_names(spans)}"
    return matches[0]


def _assert_parent(spans: list[ReadableSpan], child: ReadableSpan, parent: ReadableSpan) -> None:
    assert child.parent is not None, f"{child.name} has no parent"
    assert child.parent.span_id == parent.context.span_id, (
        f"{child.name} parent is not {parent.name}"
    )


async def test_spans_nest_from_eval_run_to_tools(scenario_dir: Path, tmp_path: Path,
                                                 span_exporter: InMemorySpanExporter) -> None:
    scenarios = _one_scenario(scenario_dir, "happy-calc-add")
    cfg = RunConfig(output_dir=tmp_path / "out", report_formats=[])
    run = await run_suite("suite-happy-calc-add", scenarios, cfg)
    assert run.passed == 1

    spans = list(span_exporter.get_finished_spans())
    names = _names(spans)
    assert any(n.startswith("eval.run") for n in names)
    assert any(n.startswith("eval.scenario") for n in names)
    assert any(n.startswith("agent.task") for n in names)
    assert "tool.calculator" in names
    assert "llm.call" in names
    assert "policy.check" in names

    run_span = next(s for s in spans if s.name.startswith("eval.run"))
    scenario_span = next(s for s in spans if s.name.startswith("eval.scenario"))
    task_span = next(s for s in spans if s.name.startswith("agent.task"))
    tool_span = _first(spans, "tool.calculator")
    llm_span = _first(spans, "llm.call")
    policy_span = _first(spans, "policy.check")

    _assert_parent(spans, scenario_span, run_span)
    _assert_parent(spans, task_span, scenario_span)
    _assert_parent(spans, tool_span, task_span)
    _assert_parent(spans, llm_span, task_span)
    _assert_parent(spans, policy_span, task_span)
    assert tool_span.attributes.get("arl.status") == "ok"


async def test_tool_span_carries_fault_and_error_attributes(scenario_dir: Path, tmp_path: Path,
                                                            span_exporter: InMemorySpanExporter) -> None:
    # http_500 on call 1: the faulted attempt's span carries the fault tag and
    # error status; the retried attempt succeeds.
    scenarios = _one_scenario(scenario_dir, "fail-500-transient-recovered")
    cfg = RunConfig(output_dir=tmp_path / "out", report_formats=[])
    run = await run_suite("suite-fail-500", scenarios, cfg)
    assert run.passed == 1

    spans = list(span_exporter.get_finished_spans())
    tool_spans = [s for s in spans if s.name.startswith("tool.")]
    faulted = [s for s in tool_spans if s.attributes.get("arl.fault") == "http_500"]
    assert faulted, "expected a tool span tagged with the injected fault"
    assert any(s.attributes.get("arl.status") == "error" for s in faulted)
    assert any(s.attributes.get("arl.status") == "ok" for s in tool_spans)


async def test_retry_spans_record_target(scenario_dir: Path, tmp_path: Path,
                                         span_exporter: InMemorySpanExporter) -> None:
    scenarios = _one_scenario(scenario_dir, "fail-500-transient-recovered")
    cfg = RunConfig(output_dir=tmp_path / "out", report_formats=[])
    await run_suite("suite-retry", scenarios, cfg)

    spans = list(span_exporter.get_finished_spans())
    retry_spans = [s for s in spans if s.name == "retry"]
    assert retry_spans, "expected retry spans for the transient-failure scenario"
    assert all("arl.target" in s.attributes for s in retry_spans)
    assert any(s.attributes.get("arl.target") not in (None, "", "model") for s in retry_spans)


async def test_approval_span_pairs_request_and_result(scenario_dir: Path, tmp_path: Path,
                                                      span_exporter: InMemorySpanExporter) -> None:
    scenarios = _one_scenario(scenario_dir, "authz-external-approval-granted")
    cfg = RunConfig(output_dir=tmp_path / "out", report_formats=[])
    await run_suite("suite-approval", scenarios, cfg)

    spans = list(span_exporter.get_finished_spans())
    approvals = [s for s in spans if s.name == "approval"]
    assert approvals, "expected an approval span"
    assert any(s.attributes.get("arl.decision") == "approve" for s in approvals)


async def test_fault_spans_named_after_fault(scenario_dir: Path, tmp_path: Path,
                                             span_exporter: InMemorySpanExporter) -> None:
    scenarios = _one_scenario(scenario_dir, "fail-500-transient-recovered")
    cfg = RunConfig(output_dir=tmp_path / "out", report_formats=[])
    await run_suite("suite-fault-span", scenarios, cfg)

    spans = list(span_exporter.get_finished_spans())
    assert any(s.name == "fault.http_500" for s in spans)


async def test_close_abandoned_ends_open_agent_span() -> None:
    exporter = InMemorySpanExporter()
    setup_tracing(exporter_instance=exporter)
    try:
        tracker = EventSpanTracker()
        event = Event(
            seq=0, t_offset_ms=0.0, wall_time=0.0, type=EventType.AGENT_START,
            run_id="run-test", scenario_id="scenario-x", agent_id="reference-agent:guarded",
            payload={"task": "never completes"},
        )
        await tracker(event)
        # The agent never completes; the runner calls close_abandoned() so the
        # task span is not lost.
        tracker.close_abandoned()
        spans = list(exporter.get_finished_spans())
        assert any(s.name == "agent.task" for s in spans)
    finally:
        setup_tracing("none")
