"""Regression coverage for verifier integrity, calibration, and Inspect export."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from arl.cli import cli
from arl.core.rlvr import OutputProbe, snapshot_test_files, verify
from arl.core.trace import Trace
from arl.evaluation.calibration import calibrate, rogan_gladen


def _trace(*, validate_second: bool = True) -> Trace:
    rows = [
        ("tool_request", "lookup", "1", {}),
        ("policy_check", "lookup", "1", {"decision": "allowed"}),
        ("tool_result", "lookup", "1", {"status": "error"}),
        ("tool_request", "lookup", "2", {}),
        ("policy_check", "lookup", "2", {"decision": "allowed"}),
        ("tool_result", "lookup", "2", {"status": "ok"}),
        ("tool_request", "check", "3", {}),
        ("policy_check", "check", "3" if validate_second else "2", {"decision": "allowed"}),
        ("tool_result", "check", "3", {"status": "ok"}),
    ]
    return Trace.model_validate({
        "run_id": "run", "scenario_id": "scenario",
        "events": [
            {"seq": i, "t_offset_ms": i, "wall_time": i, "run_id": "run", "scenario_id": "scenario",
             "name": name, "type": event, "tool_call_id": call_id, "payload": payload}
            for i, (event, name, call_id, payload) in enumerate(rows)
        ],
    })


def test_rlvr_separates_ordered_process_outcome_and_integrity() -> None:
    reward = verify(_trace(), required_steps=["lookup", "check"], outcome_passed=False)
    assert [step.reward for step in reward.steps] == [1, 1]
    assert reward.outcome_reward == 0
    assert reward.total_reward == 0.5
    reverse = verify(_trace(), required_steps=["check", "lookup"], outcome_passed=True)
    assert [step.reward for step in reverse.steps] == [1, 0]
    assert reverse.total_reward == 0.75
    skipped = verify(_trace(validate_second=False), required_steps=["lookup", "check"], outcome_passed=True)
    assert skipped.integrity_failures == ("skipped_tool_validation",)
    assert skipped.total_reward == 0


def test_rlvr_protects_test_files_and_detects_constant_output(tmp_path: Path) -> None:
    test_file = tmp_path / "test_example.py"
    test_file.write_text("assert True")
    before = snapshot_test_files(tmp_path)
    test_file.write_text("assert False")
    reward = verify(
        _trace(), required_steps=["lookup"], outcome_passed=True,
        tests_before=before, tests_after=snapshot_test_files(tmp_path),
        output_probes=[OutputProbe("first", "same"), OutputProbe("second", "same")],
    )
    assert reward.integrity_failures == ("test_files_changed", "constant_stub_output")
    assert reward.total_reward == 0
    legitimate = verify(
        _trace(), required_steps=[], outcome_passed=True,
        tests_before=before, tests_after=before,
        output_probes=[OutputProbe("same", "same"), OutputProbe("same", "same")],
    )
    assert legitimate.total_reward == 1


def test_calibration_and_bias_correction() -> None:
    gold = [True] * 1000 + [False] * 1000
    judged = [True] * 800 + [False] * 200 + [False] * 900 + [True] * 100
    calibration = calibrate(gold=gold, judged=judged)
    assert calibration.kappa == pytest.approx(0.7)
    assert calibration.tpr == 0.8
    assert calibration.tnr == 0.9
    rate = rogan_gladen(calibration, judged_passes=450, total=1000)
    assert rate.corrected == pytest.approx(0.5)
    assert rate.interval is not None
    assert 0 < rate.interval[0] < 0.5 < rate.interval[1] < 1


def test_calibration_undefined_and_uninformative_cases() -> None:
    assert calibrate(gold=[], judged=[]).kappa is None
    with pytest.raises(ValueError, match="equal length"):
        calibrate(gold=[True], judged=[])
    one_class = calibrate(gold=[True], judged=[True])
    assert rogan_gladen(one_class, judged_passes=1, total=1).interval is None
    chance = calibrate(gold=[True, False], judged=[True, True])
    rate = rogan_gladen(chance, judged_passes=1, total=2)
    assert rate.corrected is None
    assert rate.interval == (0, 1)


def test_inspect_export_covers_catalog(tmp_path: Path) -> None:
    output = tmp_path / "inspect.jsonl"
    suite = Path(__file__).resolve().parents[1] / "suites" / "full.yaml"
    result = CliRunner().invoke(cli, ["export-inspect", str(suite), "--output", str(output)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({row["id"] for row in rows}) == 92
    categories = [row["metadata"]["arl_scenario"]["category"] for row in rows]
    assert len(set(categories)) == 17
    assert categories.count("reward-hacking-and-verifier-integrity") == 8
    history = next(row for row in rows if isinstance(row["input"], list))
    scenario = history["metadata"]["arl_scenario"]
    assert [message["content"] for message in history["input"]] == [*scenario["prior_turns"], scenario["task"]]
