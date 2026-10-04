"""Example: plugging a custom agent into ARL.

An adapter is the only integration surface a framework needs to implement.
Every tool call the agent makes MUST go through ``self._env.call_tool`` — that
is where ARL injects faults, enforces the authorization policy, records the
trace and simulates human approval. Adapters never execute tools themselves.

Run it::

    python examples/custom_adapter.py            # runs one scenario offline
    python examples/custom_adapter.py --weak     # deliberately badly-behaved
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

from arl.adapters.base import AgentAdapter, AgentState, AgentStatus, TaskInput
from arl.environment.env import ToolEnvironment
from arl.recording.recorder import TraceRecorder
from arl.runner import RunConfig, run_suite
from arl.scenarios.loader import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]


class MinimalAgent(AgentAdapter):
    """A tiny reactive agent: if the task mentions an entity, look it up;
    otherwise answer from the task text. Deliberately simplistic — ARL will
    happily measure *how badly* it does."""

    def __init__(self, agent_id: str, recorder: TraceRecorder, env: ToolEnvironment,
                 task_input: TaskInput, *, naive_retry: bool = False) -> None:
        super().__init__(agent_id, recorder, env, task_input)
        self._naive_retry = naive_retry
        self._state = AgentState(status=AgentStatus.idle)

    async def start_task(self) -> None:
        await self.emit_start()
        self._state.status = AgentStatus.running
        findings: list[str] = []
        for manifest in self._task.tools:
            if manifest.risk_class.value not in ("read", "write"):
                continue  # never touch effects — security by omission
            key_field = next(iter(manifest.args_schema), "id")
            entity = next((w for w in self._task.task.split() if "-" in w and w[0].isupper()), None)
            if entity is None:
                break
            try:
                result = await self._env.call_tool(manifest.name, {key_field: entity},
                                                   tool_call_id=f"{manifest.name}-1")
                findings.append(f"{manifest.name}: {result}")
            except Exception as exc:
                if self._naive_retry:
                    # Anti-pattern: blind retry even on permanent errors.
                    with contextlib.suppress(Exception):
                        await self._env.call_tool(manifest.name, {key_field: entity},
                                                  tool_call_id=f"{manifest.name}-2")
                findings.append(f"{manifest.name}: {exc}")
        await self.emit_complete("; ".join(findings) or "nothing to report", status="success")
        self._state.status = AgentStatus.completed

    async def send_message(self, message: str) -> None:
        self._state.memory.setdefault("messages", []).append(message)

    async def receive_event(self, event: dict) -> None:
        return

    async def cancel(self) -> None:
        self._cancelled = True

    def get_state(self) -> AgentState:
        return self._state


def register() -> None:
    """Register the example adapter under the spec ``example[:naive]``."""
    from arl.adapters.registry import agent_id_of, register_agent

    def create(spec: str, *, recorder: TraceRecorder, env: ToolEnvironment,
               task_input: TaskInput, agent_config: dict | None = None) -> AgentAdapter:
        profile = spec.split(":", 1)[1] if ":" in spec else "careful"
        return MinimalAgent(
            f"example-agent:{profile}", recorder, env, task_input,
            naive_retry=(profile == "naive"),
        )

    def agent_id(spec: str) -> str:
        profile = spec.split(":", 1)[1] if ":" in spec else "careful"
        return f"example-agent:{profile}"

    register_agent("example", create, agent_id)
    assert agent_id_of("example:naive") == "example-agent:naive"


async def main() -> int:
    scenario_path = REPO_ROOT / "scenarios" / "fail-500-transient-recovered.yaml"
    scenario = load_scenario(scenario_path)
    register()

    cfg = RunConfig(agent_spec=sys.argv[1] if "--weak" not in sys.argv else "example:naive",
                    output_dir=Path(".arl/example"), report_formats=["json"])
    run = await run_suite("example", [scenario], cfg)
    result = run.results[0]
    print(f"\nscenario  : {result.scenario_id}")
    print(f"status    : {result.status} (final_status={result.final_status})")
    for check in result.checks:
        if check.status.value == "fail":
            print(f"  FAIL {check.metric}: {check.message}")
    print("\ntrace:")
    print(result.trace_text)
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
