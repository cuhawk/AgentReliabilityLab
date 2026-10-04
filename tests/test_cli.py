"""CLI tests: exit codes, validation, comparison gating, and the HITL flows
through the real runner. Everything runs offline on the deterministic agent."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from tests.conftest import REPO_ROOT

from arl.cli import cli

SCENARIOS = REPO_ROOT / "scenarios"


def test_version() -> None:
    result = CliRunner().invoke(cli, ["version"])
    assert result.exit_code == 0
    assert result.output.startswith("arl ")


def test_scenarios_validate_exit_codes() -> None:
    runner = CliRunner()
    ok = runner.invoke(cli, ["scenarios", "validate", str(SCENARIOS / "happy-calc-add.yaml")])
    assert ok.exit_code == 0

    bad_file = Path(SCENARIOS.parent / ".arl" / "bad-scenario.yaml")
    bad_file.parent.mkdir(exist_ok=True)
    bad_file.write_text(
        "scenario:\n  id: bad\n  task: t\n  expect:\n    required_tools: [ghost]\n",
        encoding="utf-8",
    )
    bad = runner.invoke(cli, ["scenarios", "validate", str(bad_file)])
    assert bad.exit_code == 1


def test_eval_exit_zero_on_pass_and_one_on_fail() -> None:
    runner = CliRunner()
    out = Path(".arl/cli-pass")
    ok = runner.invoke(cli, [
        "eval", str(SCENARIOS / "happy-calc-add.yaml"),
        "--agent", "reference:guarded", "--output-dir", str(out), "--formats", "json",
    ])
    assert ok.exit_code == 0, ok.output

    fail = runner.invoke(cli, [
        "eval", str(SCENARIOS / "demo-hallucinated-customer.yaml"),
        "--agent", "reference:weak", "--output-dir", str(Path(".arl/cli-fail")), "--formats", "json",
    ])
    assert fail.exit_code == 1, fail.output

    no_fail = runner.invoke(cli, [
        "eval", str(SCENARIOS / "demo-hallucinated-customer.yaml"),
        "--agent", "reference:weak", "--output-dir", str(Path(".arl/cli-nofail")),
        "--formats", "json", "--no-fail",
    ])
    assert no_fail.exit_code == 0, no_fail.output


def test_compare_gate_exits_one_on_regression(tmp_path: Path) -> None:
    runner = CliRunner()
    fmt = ["--formats", "json"]

    baseline_run = runner.invoke(cli, [
        "eval", str(REPO_ROOT / "suites" / "demo.yaml"),
        "--agent", "reference:guarded", "--no-fail",
        "--output-dir", str(tmp_path / "baseline"), *fmt,
    ])
    assert baseline_run.exit_code == 0, baseline_run.output
    baseline_report = _last_report(tmp_path / "baseline")

    candidate_run = runner.invoke(cli, [
        "eval", str(REPO_ROOT / "suites" / "demo.yaml"),
        "--agent", "reference:weak", "--no-fail",
        "--output-dir", str(tmp_path / "candidate"), *fmt,
    ])
    assert candidate_run.exit_code == 0  # --no-fail: we only want the report
    candidate_report = _last_report(tmp_path / "candidate")

    gate = runner.invoke(cli, [
        "compare", str(baseline_report), str(candidate_report),
    ])
    assert gate.exit_code == 1, gate.output  # weak vs guarded must be a regression
    assert "REGRESSION" in gate.output

    # Same baseline against itself → pass.
    clean = runner.invoke(cli, ["compare", str(baseline_report), str(baseline_report)])
    assert clean.exit_code == 0, clean.output


def _last_report(out_dir: Path) -> Path:
    runs = sorted(p for p in out_dir.iterdir() if p.is_dir())
    report = runs[-1] / "report.json"
    assert json.loads(report.read_text(encoding="utf-8"))["scenarios"]
    return report


def test_hitl_flows_through_the_cli() -> None:
    runner = CliRunner()
    for scenario_id in ("hitl-deny-stops", "hitl-delete-approved-fullflow",
                        "hitl-timeout-aborts", "hitl-prepare-only-no-execute",
                        "hitl-transfer-approved", "authz-external-approval-granted"):
        result = runner.invoke(cli, [
            "eval", str(SCENARIOS / f"{scenario_id}.yaml"),
            "--agent", "reference:guarded",
            "--output-dir", str(Path(".arl/cli-hitl")), "--formats", "json",
        ])
        assert result.exit_code == 0, f"{scenario_id}: {result.output}"


def test_demo_command_runs_offline() -> None:
    result = CliRunner().invoke(cli, ["demo", "--output-dir", str(Path(".arl/cli-demo"))])
    assert result.exit_code == 0, result.output
    # The demo prints both runs and a comparison table with a verdict.
    assert "reference-agent:weak" in result.output
    assert "reference-agent:guarded" in result.output
    assert "Verdict:" in result.output
