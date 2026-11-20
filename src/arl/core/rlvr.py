"""Deterministic rewards over trusted runner evidence.

The verifier and its expectations belong to the evaluator. Capture filesystem
snapshots outside the agent's control and use probes with different gold outputs.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from arl.core.events import EventType
from arl.core.trace import Trace


@dataclass(frozen=True)
class OutputProbe:
    expected: Any
    observed: Any


@dataclass(frozen=True)
class StepReward:
    tool: str
    reward: int


@dataclass(frozen=True)
class VerificationReward:
    steps: tuple[StepReward, ...]
    outcome_reward: int
    integrity_failures: tuple[str, ...]

    @property
    def process_reward(self) -> float:
        return sum(step.reward for step in self.steps) / len(self.steps) if self.steps else 0.0

    @property
    def total_reward(self) -> float:
        if self.integrity_failures:
            return 0.0
        if not self.steps:
            return float(self.outcome_reward)
        return (self.process_reward + self.outcome_reward) / 2


def snapshot_test_files(root: Path) -> dict[str, str]:
    """Snapshot contents and symlink targets under a protected test directory."""
    if root.is_symlink():
        return {".": f"symlink:{root.readlink()}"}
    snapshot = {".": "directory"} if root.is_dir() else {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = f"symlink:{path.readlink()}"
        elif path.is_file() and "__pycache__" not in path.parts:
            snapshot[relative] = sha256(path.read_bytes()).hexdigest()
    return snapshot


def verify(
    trace: Trace,
    *,
    required_steps: Sequence[str],
    outcome_passed: bool,
    tests_before: Mapping[str, str] | None = None,
    tests_after: Mapping[str, str] | None = None,
    output_probes: Sequence[OutputProbe] = (),
) -> VerificationReward:
    """Reward successful ordered tool steps and a caller-supplied deterministic outcome.

    Validation is a correlated, allowed policy decision between request and
    result. Failed attempts may be retried. Each successful call earns credit
    for at most one step and missing earlier steps block later step credit.
    """
    if (tests_before is None) != (tests_after is None):
        raise ValueError("both test snapshots are required when checking test files")

    failures = []
    validation: dict[tuple[str | None, str | None], list[int]] = {}
    for event in trace.policy_checks:
        if event.payload.get("decision") == "allowed":
            validation.setdefault((event.tool_call_id, event.name), []).append(event.seq)

    request_counts = Counter(event.tool_call_id for event in trace.by_type(EventType.TOOL_REQUEST))
    result_counts = Counter(event.tool_call_id for event in trace.by_type(EventType.TOOL_RESULT))
    malformed_success = any(
        not event.tool_call_id or request_counts[event.tool_call_id] != 1 or result_counts[event.tool_call_id] != 1
        for event in trace.by_type(EventType.TOOL_RESULT)
        if event.payload.get("status") == "ok"
    )
    completed = 0
    calls = sorted(trace.tool_calls, key=lambda call: call.result.seq if call.result else float("inf"))
    for call in calls:
        if not call.ok or call.result is None:
            continue
        result = call.result
        if (
            request_counts[call.tool_call_id] != 1
            or result_counts[call.tool_call_id] != 1
            or result.name != call.name
            or not any(
                call.request.seq < seq < result.seq
                for seq in validation.get((call.tool_call_id, call.name), ())
            )
        ):
            malformed_success = True
            continue
        if completed < len(required_steps) and call.name == required_steps[completed]:
            completed += 1

    if malformed_success:
        failures.append("skipped_tool_validation")
    if tests_before != tests_after:
        failures.append("test_files_changed")
    if output_probes:
        first = output_probes[0]
        if any(probe.expected != first.expected for probe in output_probes) and all(
            probe.observed == first.observed for probe in output_probes
        ):
            failures.append("constant_stub_output")

    steps = tuple(StepReward(tool, int(index < completed)) for index, tool in enumerate(required_steps))
    return VerificationReward(steps, int(outcome_passed), tuple(failures))
