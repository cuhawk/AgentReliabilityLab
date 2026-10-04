"""``arl`` command line interface.

Typical usage::

    arl eval suites/ci.yaml --agent reference:guarded
    arl eval suites/ci.yaml --baseline runs/.../report.json --fail-on-regression
    arl compare baseline/report.json candidate/report.json
    arl demo
    arl serve
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
from click import group

from arl import __version__


def yaml_safe_load(path: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _repo_root() -> Path:
    """Repo root when running from a checkout; falls back to CWD."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "suites").is_dir():
            return candidate
    return Path.cwd()


def _load_suite_or_scenario(path: str) -> tuple[str, list]:
    from arl.scenarios.loader import ScenarioError, load_scenario
    from arl.scenarios.suite import load_suite

    file = Path(path)
    if not file.is_file():
        raise ScenarioError(f"not found: {file}")
    raw = yaml_safe_load(str(file))
    if "suite" in raw:
        suite, scenarios = load_suite(file)
        return suite.id, scenarios
    scenario = load_scenario(file)
    return scenario.id, [scenario]


def _print_summary(run: Any) -> None:
    click.echo(f"\nSuite '{run.suite_id}' · agent '{run.agent_id}' · run {run.run_id}")
    click.echo(f"Passed {run.passed}/{len(run.results)} scenarios")
    for name, m in run.metrics.items():
        ci = ""
        if m.ci_low is not None:
            ci = f"  CI95 [{m.ci_low:.3f}, {m.ci_high:.3f}]"
        click.echo(f"  {name:36} {m.value:>12.4f}{ci}")
    failures = [r for r in run.results if r.status != "pass"]
    if failures:
        click.echo("\nFailed scenarios:")
        for r in failures:
            click.echo(f"  {r.scenario_id}:")
            for c in r.failed_checks:
                msg = f" — {c.message}" if c.message else ""
                click.echo(f"    [{c.metric}] {msg}")
    for fmt, path in run.report_paths.items():
        click.echo(f"report[{fmt}]: {path}")


@group()
def cli() -> None:
    """AgentReliabilityLab: measurable, recoverable, secure agents."""


@cli.command()
@click.argument("suite_path")
@click.option("--agent", default="reference:guarded", show_default=True,
              help="Agent spec: reference[:guarded|weak] | openai:<model> | http:<url>")
@click.option("--agent-config", "agent_config_path", type=click.Path(exists=True), default=None)
@click.option("--output-dir", default=".arl/runs", show_default=True)
@click.option("--db", "db_path", default=None, help="SQLite database path (enables persistence).")
@click.option("--formats", default="json,md,html", show_default=True)
@click.option("--trace-exporter", default="none", type=click.Choice(["none", "console", "otlp"]))
@click.option("--category", "categories", multiple=True, help="Only run these categories.")
@click.option("--tag", "tags", multiple=True, help="Only scenarios with these tags.")
@click.option("--max-scenarios", type=int, default=None)
@click.option("--model", default=None, help="Override model id used for cost calculation.")
@click.option("--baseline", type=click.Path(exists=True), default=None,
              help="Baseline report.json to compare against.")
@click.option("--thresholds", "thresholds_path", type=click.Path(exists=True), default=None)
@click.option("--fail-on-regression/--no-fail-on-regression", default=True, show_default=True)
@click.option("--no-fail", is_flag=True, help="Exit 0 even when scenarios fail (demo mode).")
def eval(
    suite_path: str,
    agent: str,
    agent_config_path: str | None,
    output_dir: str,
    db_path: str | None,
    formats: str,
    trace_exporter: str,
    categories: tuple[str, ...],
    tags: tuple[str, ...],
    max_scenarios: int | None,
    model: str | None,
    baseline: str | None,
    thresholds_path: str | None,
    fail_on_regression: bool,
    no_fail: bool,
) -> None:
    """Run an eval suite (or a single scenario file) and write reports."""
    from arl.runner import RunConfig, run_suite
    from arl.scenarios.loader import ScenarioError

    try:
        suite_id, scenarios = _load_suite_or_scenario(suite_path)
    except ScenarioError as exc:
        raise click.ClickException(str(exc)) from exc

    if categories:
        scenarios = [s for s in scenarios if s.category.value in categories]
    if tags:
        scenarios = [s for s in scenarios if set(tags) & set(s.tags)]
    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]
    if not scenarios:
        raise click.ClickException("no scenarios matched the given filters")

    agent_config: dict[str, Any] = {}
    if agent_config_path:
        agent_config = yaml_safe_load(agent_config_path)

    cfg = RunConfig(
        agent_spec=agent,
        agent_config=agent_config,
        output_dir=Path(output_dir),
        db_path=Path(db_path) if db_path else None,
        report_formats=[f.strip() for f in formats.split(",") if f.strip()],
        trace_exporter=trace_exporter,
        model=model,
    )
    if cfg.trace_exporter != "none":
        from arl.observability.tracing import setup_tracing

        setup_tracing(cfg.trace_exporter)

    storage = None
    if cfg.db_path is not None:
        from arl.recording.storage import SqliteStorage

        storage = SqliteStorage(cfg.db_path)

    baseline_report = None
    thresholds = None
    if baseline:
        baseline_report = json.loads(Path(baseline).read_text(encoding="utf-8"))
        from arl.regression.thresholds import load_thresholds

        thresholds = load_thresholds(thresholds_path)

    run = asyncio.run(run_suite(
        suite_id, scenarios, cfg, storage=storage,
        baseline_report=baseline_report, thresholds=thresholds,
    ))
    _print_summary(run)

    exit_code = 0
    if run.failed > 0 and not no_fail:
        exit_code = 1
    if baseline_report is not None and run.comparison:
        verdict = run.comparison["verdict"]
        click.echo(f"\nBaseline comparison verdict: {verdict.upper()}")
        for row in run.comparison["rows"]:
            if row["status"] == "regression":
                click.echo(f"  REGRESSION {row['metric']}: {row['note']}")
        if verdict == "regression" and fail_on_regression:
            exit_code = 1
    if storage is not None:
        storage.close()
    sys.exit(exit_code)


@cli.command()
@click.argument("baseline_path", type=click.Path(exists=True))
@click.argument("candidate_path", type=click.Path(exists=True))
@click.option("--thresholds", "thresholds_path", type=click.Path(exists=True), default=None)
@click.option("--db", "db_path", default=None, help="Store the comparison in SQLite.")
def compare(baseline_path: str, candidate_path: str, thresholds_path: str | None, db_path: str | None) -> None:
    """Compare two report.json files; exit 1 on regression (CI gate)."""
    from arl.evaluation.aggregates import MetricValue
    from arl.regression.compare import compare_metrics, compare_scenarios
    from arl.regression.thresholds import load_thresholds

    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    thresholds = load_thresholds(thresholds_path)

    base_metrics = {name: MetricValue.model_validate(m) for name, m in baseline.get("metrics", {}).items()}
    cand_metrics = {name: MetricValue.model_validate(m) for name, m in candidate.get("metrics", {}).items()}
    comparison = compare_metrics(
        base_metrics, cand_metrics, thresholds,
        baseline_run_id=str(baseline.get("run_id", "")),
        candidate_run_id=str(candidate.get("run_id", "")),
    )
    comparison.scenario_rows = compare_scenarios(
        baseline.get("scenarios", []), candidate.get("scenarios", [])
    )

    click.echo(comparison.render_table())
    click.echo(f"\nVerdict: {comparison.verdict.upper()}")
    if db_path:
        from arl.recording.storage import SqliteStorage

        storage = SqliteStorage(db_path)
        storage.save_comparison(comparison.model_dump())
        storage.close()
        click.echo(f"Stored comparison {comparison.comparison_id} in {db_path}")
    sys.exit(0 if comparison.verdict == "pass" else 1)


@cli.group()
def scenarios() -> None:
    """Inspect and validate scenario files."""


@scenarios.command("validate")
@click.argument("paths", nargs=-1, required=True)
def validate_scenarios(paths: tuple[str, ...]) -> None:
    """Validate scenario/suite files without running them."""
    from arl.scenarios.loader import ScenarioError, load_scenarios

    failures = 0
    for path in paths:
        try:
            for s in load_scenarios([path]):
                click.echo(click.style(f"OK   {s.id} ({s.category.value})", fg="green"))
        except ScenarioError as exc:
            failures += 1
            click.echo(click.style(f"FAIL {exc}", fg="red"))
    sys.exit(1 if failures else 0)


@scenarios.command("list")
@click.argument("paths", nargs=-1, required=True)
def list_scenarios(paths: tuple[str, ...]) -> None:
    """List scenarios with category and fault summary."""
    from arl.scenarios.loader import ScenarioError, load_scenarios

    try:
        loaded = load_scenarios(list(paths))
    except ScenarioError as exc:
        raise click.ClickException(str(exc)) from exc
    for s in loaded:
        faults = [f.fault.value for spec in s.tools.values() for f in spec.faults]
        faults += [f.fault.value for f in s.model_faults]
        fault_str = f" faults={','.join(faults)}" if faults else ""
        click.echo(f"{s.id:44} {s.category.value:20}{fault_str}")


@cli.command()
@click.option("--output-dir", default=".arl/demo", show_default=True)
@click.option("--db", "db_path", default=None)
def demo(output_dir: str, db_path: str | None) -> None:
    """One-command demo: weak agent vs guarded agent on the demo suite, then a
    regression comparison (baseline=weak, candidate=guarded). Fully offline."""
    from arl.evaluation.aggregates import MetricValue
    from arl.regression.compare import compare_metrics, compare_scenarios
    from arl.regression.thresholds import load_thresholds
    from arl.runner import RunConfig, run_suite
    from arl.scenarios.loader import ScenarioError
    from arl.scenarios.suite import load_suite

    suite_file = _repo_root() / "suites" / "demo.yaml"
    try:
        _, scenarios = load_suite(suite_file)
    except ScenarioError as exc:
        raise click.ClickException(str(exc)) from exc

    out = Path(output_dir)
    thresholds = load_thresholds(None)
    storage = None
    if db_path:
        from arl.recording.storage import SqliteStorage

        storage = SqliteStorage(db_path)

    runs: list[Any] = []
    for profile in ("reference:weak", "reference:guarded"):
        cfg = RunConfig(agent_spec=profile, output_dir=out, report_formats=["json", "md", "html"])
        run = asyncio.run(run_suite("demo", scenarios, cfg, storage=storage))
        _print_summary(run)
        runs.append(run)

    baseline = json.loads(Path(runs[0].report_paths["json"]).read_text(encoding="utf-8"))
    comparison = compare_metrics(
        {name: MetricValue.model_validate(m) for name, m in baseline["metrics"].items()},
        runs[1].metrics, thresholds,
        baseline_run_id=runs[0].run_id, candidate_run_id=runs[1].run_id,
    )
    comparison.scenario_rows = compare_scenarios(
        baseline.get("scenarios", []),
        [{"scenario_id": r.scenario_id, "status": r.status} for r in runs[1].results],
    )
    click.echo("\nBaseline = reference:weak, Candidate = reference:guarded\n")
    click.echo(comparison.render_table())
    click.echo(f"\nVerdict: {comparison.verdict.upper()}")
    if storage is not None:
        storage.save_comparison(comparison.model_dump())
        storage.close()
    click.echo("Full reports and traces: " + str(out))


@cli.command()
@click.option("--db", "db_path", default=".arl/lab.db", show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8600, show_default=True)
def serve(db_path: str, host: str, port: int) -> None:
    """Run the dashboard web UI over a stored database."""
    import uvicorn

    from arl.dashboard.app import create_app

    uvicorn.run(create_app(db_path), host=host, port=port, log_level="info")


def serve_command_entry() -> None:
    """Console-script entry for ``arl-dashboard`` (same as ``arl serve``)."""
    cli(["serve"])


@cli.command()
@click.argument("scenario_path", type=click.Path(exists=True))
def mcp_serve(scenario_path: str) -> None:
    """Serve a scenario's tools as an MCP server on stdio.

    MCP-native agents connect to this process (tools/list, tools/call); every
    call crosses the ARL gateway, so faults, policy and recording apply.
    """
    from arl.adapters.mcp_gateway import serve_stdio
    from arl.environment.env import ToolEnvironment
    from arl.faults.injector import FaultInjector
    from arl.recording.recorder import TraceRecorder
    from arl.scenarios.loader import ScenarioError

    try:
        _, scenarios = _load_suite_or_scenario(scenario_path)
    except ScenarioError as exc:
        raise click.ClickException(str(exc)) from exc
    scenario = scenarios[0]

    recorder = TraceRecorder(f"mcp-{scenario.id}", scenario.id, "mcp-gateway")
    injector = FaultInjector(scenario, recorder)
    env = ToolEnvironment(scenario, recorder, injector)
    click.echo(f"mcp: serving {len(scenario.tools)} tools from scenario '{scenario.id}'", err=True)
    serve_stdio(env)
    trace = asyncio.run(recorder.finish("completed"))
    click.echo(f"mcp: session recorded {len(trace.events)} events", err=True)


@cli.command()
def version() -> None:
    """Print the ARL version."""
    click.echo(f"arl {__version__}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
