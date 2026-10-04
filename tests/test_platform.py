"""Platform tests: fault injection determinism, policy engine, canary secrets,
cost accounting, scenario loading, storage roundtrip, reports, and the
regression gate. Everything CPU-only and deterministic — no RNG, no GPU."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from arl.core.events import EventType, RiskClass
from arl.core.trace import Trace
from arl.cost import cost_for_trace, estimate_tokens, price_for
from arl.environment.policy import PolicyEngine, approval_key
from arl.environment.secrets import build_secret_map, substitute_secret_refs
from arl.evaluation.aggregates import MetricValue
from arl.faults.injector import FaultInjector
from arl.recording.recorder import TraceRecorder
from arl.recording.storage import SqliteStorage
from arl.scenarios.loader import ScenarioError
from arl.scenarios.schema import ToolSpec

# ------------------------------------------------------------------ faults

def _scenario_with_faults():
    from arl.scenarios.schema import (
        EnvSpec,
        FaultStep,
        FaultType,
        Scenario,
        ToolSpec,
    )
    return Scenario(
        id="fault-unit", task="t",
        tools={
            "a": ToolSpec(name="a", kind="data",
                          args_schema={"id": {"type": "string"}},
                          params={"records": []},
                          faults=[FaultStep(call=1, fault=FaultType.http_500),
                                  FaultStep(every=2, fault=FaultType.timeout)]),
            "b": ToolSpec(name="b", kind="data",
                          args_schema={"id": {"type": "string"}},
                          params={"records": []}),
        },
        model_faults=[FaultStep(call=2, fault=FaultType.backend_unavailable),
                      FaultStep(call=1, fault=FaultType.truncated_response)],
        state_faults=[FaultStep(call=1, fault=FaultType.missing_state)],
        run_faults=[FaultStep(call=2, fault=FaultType.interrupted)],
        env=EnvSpec(),
    )


def test_tool_fault_schedule_is_ordinal_keyed() -> None:
    scenario = _scenario_with_faults()

    async def run() -> list[str]:
        recorder = TraceRecorder("r", "s", "a")
        injector = FaultInjector(scenario, recorder)
        taken = []
        for _ in range(4):  # a: 500, timeout, - , timeout (every=2 → ordinals 2,4)
            step = await injector.take_tool_fault("a")
            taken.append(step.fault.value if step else "-")
        step_b = await injector.take_tool_fault("b")  # separate counter → nothing
        taken.append(step_b.fault.value if step_b else "-")
        return taken

    assert asyncio.run(run()) == ["http_500", "timeout", "-", "timeout", "-"]
    faults = [e for e in asyncio.run(_replay_events()) if e.type is EventType.FAULT_INJECTED]
    assert faults  # FAULT_INJECTED events were recorded


async def _replay_events():
    recorder = TraceRecorder("r", "s", "a")
    injector = FaultInjector(_scenario_with_faults(), recorder)
    await injector.take_tool_fault("a")
    await injector.take_tool_fault("a")
    return recorder.events


def test_model_fault_counters_split_availability_from_content() -> None:
    scenario = _scenario_with_faults()

    async def run() -> list[str]:
        recorder = TraceRecorder("r", "s", "a")
        injector = FaultInjector(scenario, recorder)
        taken = []
        # Call 1, not a content call: the content fault is skipped, the
        # availability fault is scheduled for ordinal 2 → nothing yet.
        taken.append(await injector.take_model_fault(content_call=False))
        # Call 2 (a content call): the availability fault fires first
        # (first scheduled match wins).
        taken.append(await injector.take_model_fault(content_call=True))
        return [t.fault.value if t else "-" for t in taken]

    assert asyncio.run(run()) == ["-", "backend_unavailable"]


def test_run_fault_lookup_does_not_emit() -> None:
    scenario = _scenario_with_faults()

    async def run() -> tuple:
        recorder = TraceRecorder("r", "s", "a")
        injector = FaultInjector(scenario, recorder)
        fault = injector.run_fault_for(2)  # interrupted scheduled at ordinal 2
        none_at = injector.run_fault_for(1)
        return (fault.fault.value if fault else "-", none_at is None, len(recorder.events))

    fault_val, none_at_1, events = asyncio.run(run())
    assert fault_val == "interrupted"
    assert none_at_1
    assert events == 0  # lookup never emits


# ------------------------------------------------------------------- policy

def _write_tool(risk_class: RiskClass, scope=None) -> ToolSpec:
    return ToolSpec(name="t", kind="data", risk_class=risk_class,
                    args_schema={"id": {"type": "string", "required": True}},
                    params={"records": []}, scope=scope)


def test_policy_engine_decisions() -> None:
    from arl.scenarios.schema import DEFAULT_POLICY, PolicyAction, ScopeSpec

    engine = PolicyEngine(DEFAULT_POLICY)

    # READ → allow.
    assert engine.check(_write_tool(RiskClass.READ), {"id": "x"}, {}).allowed

    # PRIVILEGED → deny.
    decision = engine.check(_write_tool(RiskClass.PRIVILEGED), {"id": "x"}, {})
    assert not decision.allowed and decision.action is PolicyAction.deny

    # WRITE out of scope → denied with scope reason; in scope → allowed.
    scoped = _write_tool(RiskClass.WRITE, ScopeSpec(field="id", allowed=["C-101"]))
    out = engine.check(scoped, {"id": "C-999"}, {})
    assert not out.allowed and "outside allowed scope" in out.reason
    missing = engine.check(scoped, {}, {})
    assert not missing.allowed and "missing scope field" in missing.reason
    assert engine.check(scoped, {"id": "C-101"}, {}).allowed

    # EXTERNAL_SIDE_EFFECT requires approval, granted per (tool, args) key.
    effect = _write_tool(RiskClass.EXTERNAL_SIDE_EFFECT)
    blocked = engine.check(effect, {"id": "C-101"}, {})
    assert not blocked.allowed and "human approval required" in blocked.reason
    key = approval_key("t", {"id": "C-101"})
    granted = engine.check(effect, {"id": "C-101"}, {key: "granted"})
    assert granted.allowed and "granted" in granted.reason
    # A different arg set does NOT inherit the approval.
    assert not engine.check(effect, {"id": "C-102"}, {key: "granted"}).allowed


# ------------------------------------------------------------------ secrets

def test_canary_secrets_deterministic_and_substituted() -> None:
    from arl.scenarios.schema import EnvSpec

    env_a = EnvSpec(secrets=["API_KEY", "DB_PASSWORD"])
    env_b = EnvSpec(secrets=["API_KEY", "DB_PASSWORD"])
    secrets_a = build_secret_map(env_a, "scenario-x")
    secrets_b = build_secret_map(env_b, "scenario-x")
    other = build_secret_map(env_a, "scenario-y")

    assert secrets_a == secrets_b  # deterministic, no RNG
    assert secrets_a["API_KEY"] != secrets_a["DB_PASSWORD"]
    assert secrets_a["API_KEY"] != other["API_KEY"]  # scenario-scoped
    assert secrets_a["API_KEY"].startswith("arl-canary-")

    params = {"note": "key is {{SECRET:API_KEY}}", "plain": "x"}
    substituted = substitute_secret_refs(params, secrets_a)
    assert substituted["note"] == f"key is {secrets_a['API_KEY']}"
    assert "{{SECRET" not in json.dumps(substituted)


# --------------------------------------------------------------------- cost

def test_cost_accounting_honesty_rules() -> None:
    trace = _trace_with_usage(token_source="estimated", model="scripted")
    cost, source = cost_for_trace(trace, "scripted")
    assert cost == 0.0 and source == "estimated"  # scripted/mock is free

    trace = _trace_with_usage(token_source="reported", model=None)
    cost, source = cost_for_trace(trace, "gpt-4o")
    # 2000 input @ $2.50/Mtok + 500 output @ $10/Mtok = 0.005 + 0.005 = $0.01
    assert source == "reported"
    assert cost == pytest.approx(0.01, abs=1e-9)

    cost, source = cost_for_trace(trace, "totally-unknown-model")
    assert cost == 0.0 and source == "unpriced"  # never invent a price

    assert price_for("gpt-4o-mini") == (0.15, 0.60)
    assert estimate_tokens("x" * 40) == 10


def _trace_with_usage(*, token_source: str, model: str | None) -> Trace:

    async def run():
        recorder = TraceRecorder("r", "s", "a")
        await recorder.emit_model_request(step=0, input_tokens=10)
        await recorder.emit_model_response(step=0, input_tokens=2000, output_tokens=500,
                                           preview="answer", model=model or "scripted",
                                           token_source=token_source)
        return await recorder.finish("completed")

    return asyncio.run(run())


# ------------------------------------------------------- loader & suites

def test_library_merge_concatenates_faults(tmp_path: Path) -> None:
    from arl.scenarios.loader import load_scenarios

    (tmp_path / "_library.yaml").write_text(
        """
tools:
  lookup:
    name: lookup
    kind: data
    risk_class: read
    args_schema: {id: {type: string, required: true}}
    params: {records: [{id: R-1, value: 42}]}
    faults:
      - {call: 1, fault: http_500}
""", encoding="utf-8")
    (tmp_path / "s.yaml").write_text(
        """
scenario:
  id: lib-merge
  task: use lookup
  tools:
    lookup:
      library: lookup
      faults:
        - {call: 2, fault: timeout}
  expect:
    final_status: success
""", encoding="utf-8")

    scenarios = load_scenarios([str(tmp_path / "s.yaml")])
    scenario = scenarios[0] if isinstance(scenarios, list) else scenarios
    tool = scenario.tools["lookup"]
    assert [f.fault.value for f in tool.faults] == ["http_500", "timeout"]  # merged, in order


def test_suite_loading_with_shorthand_and_globs(tmp_path: Path) -> None:
    from arl.scenarios.suite import load_suite

    for i in range(3):
        (tmp_path / f"case-{i}.yaml").write_text(
            f"""
scenario:
  id: suite-scenario-{i}
  task: t
  expect:
    final_status: success
""", encoding="utf-8")
    (tmp_path / "suite.yaml").write_text(
        """
suite: demo-suite
scenarios: "case-*.yaml"
""", encoding="utf-8")

    suite, scenarios = load_suite(tmp_path / "suite.yaml")
    assert suite.id == "demo-suite"
    assert [s.id for s in scenarios] == [f"suite-scenario-{i}" for i in range(3)]

    with pytest.raises(ScenarioError):
        # Unknown shorthand → clear error, not silence.
        load_suite(_bad_suite(tmp_path))


def _bad_suite(tmp_path: Path) -> Path:
    bad = tmp_path / "bad-suite.yaml"
    bad.write_text("suite: bad\nscenarios: \"nothing-here/*.yaml\"\n", encoding="utf-8")
    return bad


# ----------------------------------------------------------- storage & reports

async def _run_stored(tmp_path: Path, agent_spec: str = "reference:guarded") -> SqliteStorage:
    from arl.runner import RunConfig, run_suite
    from arl.scenarios.loader import load_scenarios

    found = load_scenarios([str(_repo() / "scenarios" / "happy-calc-add.yaml")])
    scenarios = found if isinstance(found, list) else [found]
    storage = SqliteStorage(tmp_path / "lab.db")
    cfg = RunConfig(agent_spec=agent_spec, output_dir=tmp_path / "out",
                    report_formats=["json", "md", "html"])
    await run_suite("platform-suite", scenarios, cfg, storage=storage)
    return storage


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def test_storage_roundtrip(tmp_path: Path) -> None:
    storage = asyncio.run(_run_stored(tmp_path))
    try:
        runs = storage.list_runs()
        assert len(runs) == 1
        run_id = runs[0]["id"]
        run = storage.get_run(run_id)
        assert run["suite"] == "platform-suite"

        results = storage.get_scenario_results(run_id)
        assert results[0]["scenario_id"] == "happy-calc-add"
        checks = json.loads(results[0]["checks_json"])
        assert any(c["metric"] == "task_success" for c in checks)

        events = storage.get_events(run_id, "happy-calc-add")
        assert events[0]["type"] == "agent_start"
        assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)

        metrics = storage.get_scenario_metrics(run_id)
        assert any(m["name"] == "tool_calls" and m["scenario_id"] for m in metrics)
    finally:
        storage.close()


def test_reports_written_and_failure_trace_embedded(tmp_path: Path) -> None:
    from arl.runner import RunConfig, run_suite
    from arl.scenarios.loader import load_scenarios

    found = load_scenarios([str(_repo() / "scenarios" / "demo-hallucinated-customer.yaml")])
    scenarios = found if isinstance(found, list) else [found]
    cfg = RunConfig(agent_spec="reference:weak", output_dir=tmp_path / "out",
                    report_formats=["json", "md", "html"])
    run = asyncio.run(run_suite("report-suite", scenarios, cfg))

    report_path = Path(run.report_paths["json"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_version"] == 1
    assert report["failed_scenarios"], "weak should fail the hallucination scenario"
    assert report["scenarios"][0]["trace_link"].startswith("traces/")

    trace_file = report_path.parent / report["scenarios"][0]["trace_link"]
    assert trace_file.exists() and "TOOL_REQUEST" in trace_file.read_text(encoding="utf-8")

    assert Path(run.report_paths["md"]).exists()
    html = Path(run.report_paths["html"]).read_text(encoding="utf-8")
    assert "AgentReliabilityLab" in html
    assert "TOOL_REQUEST" in html  # failure traces embedded for the viewer


# --------------------------------------------------------- regression gates

def _mv(name: str, value: float, ci_low=None, ci_high=None, n=10) -> MetricValue:
    return MetricValue(name=name, value=value, ci_low=ci_low, ci_high=ci_high, n=n)


def test_regression_gates() -> None:
    from arl.regression.compare import compare_metrics
    from arl.regression.thresholds import DEFAULT_THRESHOLDS

    def compare(base: float, cand: float, metric: str = "task_success_rate"):
        return compare_metrics(
            {metric: _mv(metric, base)}, {metric: _mv(metric, cand)}, DEFAULT_THRESHOLDS,
        )

    # Small improvement → pass.
    assert compare(0.95, 0.97).verdict == "pass"
    # 4-point drop > 3-point gate → regression.
    assert compare(0.95, 0.91).verdict == "regression"
    # Sub-noise-floor delta never fires.
    row = compare(0.95, 0.948).rows[0]
    assert row.status == "pass_low_effect"

    # Hard cap: unauthorized_action_rate must stay at exactly 0.
    assert compare(0.0, 0.05, "unauthorized_action_rate").verdict == "regression"

    # Noise floor on latency: +10ms is below the 50ms min_effect.
    latency = compare(200.0, 210.0, "p95_latency_ms")
    assert latency.verdict == "pass"
    # +30% on p95 (260 → 60ms delta) exceeds 25% gate → regression.
    assert compare(200.0, 260.0, "p95_latency_ms").verdict == "regression"


def test_compare_scenarios_lists_new_failures() -> None:
    from arl.regression.compare import compare_scenarios

    rows = compare_scenarios(
        [{"scenario_id": "a", "status": "pass"}, {"scenario_id": "b", "status": "pass"}],
        [{"scenario_id": "a", "status": "pass"}, {"scenario_id": "b", "status": "fail"},
         {"scenario_id": "c", "status": "pass"}],
    )
    by_id = {r.scenario_id: r for r in rows}
    assert by_id["b"].candidate == "fail"
    assert by_id["b"].baseline == "pass"
    assert by_id["c"].baseline is None  # new scenario in candidate


# ------------------------------------------------- weak vs guarded property

def test_weak_fails_and_guarded_passes_on_demo_suite() -> None:
    from arl.runner import RunConfig, run_suite
    from arl.scenarios.suite import load_suite

    _, scenarios = load_suite(_repo() / "suites" / "demo.yaml")
    weak = asyncio.run(run_suite("demo", scenarios,
                                 RunConfig(agent_spec="reference:weak", output_dir=Path(".arl/t-weak"),
                                           report_formats=[])))
    guarded = asyncio.run(run_suite("demo", scenarios,
                                    RunConfig(agent_spec="reference:guarded",
                                              output_dir=Path(".arl/t-guarded"), report_formats=[])))
    assert weak.passed < len(weak.results), "weak profile must fail the demo suite"
    assert guarded.passed == len(guarded.results), "guarded profile must pass the demo suite"

    failed_metrics = {c.metric for r in weak.results for c in r.failed_checks}
    assert {"injection_resistance", "secret_leakage", "unauthorized_actions"} & failed_metrics
