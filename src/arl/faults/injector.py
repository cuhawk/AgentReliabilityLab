"""Fault injection: deterministic, ordinal-keyed fault scheduling.

Faults are declared in the scenario (``tools.<name>.faults``, ``model_faults``,
``state_faults``, ``run_faults``). The injector counts call ordinals per target
and returns the fault scheduled for the current call, if any. Because schedules
are keyed by ordinal, the same scenario always produces the same fault
sequence — no RNG in the scoring path.
"""

from __future__ import annotations

from arl.core.events import EventType
from arl.recording.recorder import TraceRecorder
from arl.scenarios.schema import FaultStep, FaultType, Scenario

MODEL_CONTENT_FAULTS = frozenset({FaultType.invalid_structured_output, FaultType.truncated_response})


class FaultInjector:
    def __init__(self, scenario: Scenario, recorder: TraceRecorder) -> None:
        self._scenario = scenario
        self._recorder = recorder
        self._tool_ordinals: dict[str, int] = {}
        self._model_all_ordinal = 0
        self._model_content_ordinal = 0
        self._state_ordinal = 0

    # --- tool faults -----------------------------------------------------

    async def take_tool_fault(self, tool_name: str) -> FaultStep | None:
        """Advance the ordinal for *tool_name*, emit FAULT_INJECTED if scheduled."""
        ordinal = self._tool_ordinals.get(tool_name, 0) + 1
        self._tool_ordinals[tool_name] = ordinal
        spec = self._scenario.tools.get(tool_name)
        if spec is None:
            return None
        for step in spec.faults:
            if step.applies_to(ordinal):
                await self._emit(step, name=tool_name, call_ordinal=ordinal)
                return step
        return None

    def tool_ordinal(self, tool_name: str) -> int:
        """Current (post-increment) call ordinal for a tool."""
        return self._tool_ordinals.get(tool_name, 0)

    # --- model faults ------------------------------------------------------
    # Availability faults (backend_unavailable, model_rate_limit) are counted
    # over *all* model calls. Content faults (invalid_structured_output,
    # truncated_response) are counted over *final-answer* model calls only —
    # agents cannot corrupt a tool-selection decision, so this is what scenario
    # authors mean when they schedule "the compose step".

    async def take_model_fault(self, *, content_call: bool = False) -> FaultStep | None:
        self._model_all_ordinal += 1
        taken: FaultStep | None = None
        for step in self._scenario.model_faults:
            is_content_fault = step.fault in MODEL_CONTENT_FAULTS
            if is_content_fault and not content_call:
                continue
            ordinal = self._model_content_ordinal if is_content_fault else self._model_all_ordinal
            if step.applies_to(ordinal):
                taken = step
                break
        if content_call:
            self._model_content_ordinal += 1
        if taken is not None:
            await self._emit(taken, name="model", call_ordinal=self._model_all_ordinal)
        return taken

    @property
    def has_model_faults(self) -> bool:
        return bool(self._scenario.model_faults)

    # --- state faults (applied on state reads) ------------------------------

    async def take_state_fault(self) -> FaultStep | None:
        self._state_ordinal += 1
        for step in self._scenario.state_faults:
            if step.applies_to(self._state_ordinal):
                await self._emit(step, name="state", call_ordinal=self._state_ordinal)
                return step
        return None

    # --- run faults (checked by the environment around tool calls) ----------

    def run_fault_for(self, global_tool_ordinal: int) -> FaultStep | None:
        """Return (without emitting) the run fault scheduled for this ordinal."""
        for step in self._scenario.run_faults:
            if step.applies_to(global_tool_ordinal):
                return step
        return None

    def global_tool_ordinal(self) -> int:
        return sum(self._tool_ordinals.values())

    # ----------------------------------------------------------------------

    async def _emit(self, step: FaultStep, *, name: str, call_ordinal: int) -> None:
        await self._recorder.emit(
            EventType.FAULT_INJECTED,
            name=name,
            fault=step.fault.value,
            call_ordinal=call_ordinal,
            params=dict(step.params),
            description=step.description,
        )


def is_transient(fault: FaultType | None) -> bool:
    from arl.scenarios.schema import TRANSIENT_TOOL_FAULTS

    return fault in TRANSIENT_TOOL_FAULTS
