"""Trace recording: assigns sequence numbers and offsets, buffers events,
and produces the final :class:`Trace`."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from arl.core.events import Event, EventType, RiskClass
from arl.core.trace import Trace

Observer = Callable[[Event], Awaitable[None]]


class TraceRecorder:
    """Records events for one scenario run. Single asyncio task emits through
    :meth:`emit`; observers (e.g. OpenTelemetry span exporter) are awaited so
    span parents/children stay ordered."""

    def __init__(self, run_id: str, scenario_id: str, agent_id: str) -> None:
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.agent_id = agent_id
        # perf_counter, not monotonic: on Windows monotonic ticks at ~15.6 ms,
        # which would collapse fast scenarios to 0 ms durations.
        self._t0_mono = time.perf_counter()
        self._t0_wall = time.time()
        self._seq = 0
        self.events: list[Event] = []
        self._observers: list[Observer] = []
        self._finished = False

    def add_observer(self, observer: Observer) -> None:
        self._observers.append(observer)

    def now_offset_ms(self) -> float:
        return (time.perf_counter() - self._t0_mono) * 1000.0

    async def emit(
        self,
        event_type: EventType,
        *,
        name: str | None = None,
        tool_call_id: str | None = None,
        risk_class: RiskClass | None = None,
        attempt: int = 1,
        **payload: Any,
    ) -> Event:
        event = Event(
            seq=self._seq,
            t_offset_ms=round(self.now_offset_ms(), 3),
            wall_time=time.time(),
            type=event_type,
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            agent_id=self.agent_id,
            name=name,
            tool_call_id=tool_call_id,
            risk_class=risk_class,
            attempt=attempt,
            payload=payload,
        )
        self._seq += 1
        self.events.append(event)
        for observer in self._observers:
            await observer(event)
        return event

    async def emit_retry(self, target: str, attempt: int, reason: str, backoff_ms: float = 0.0) -> Event:
        """A RETRY event: *target* is the tool name or "model"."""
        return await self.emit(EventType.RETRY, name=target, attempt=attempt, reason=reason, backoff_ms=backoff_ms)

    async def emit_error(self, error: str, kind: str = "agent_error") -> Event:
        return await self.emit(EventType.AGENT_ERROR, error=error, kind=kind)

    async def emit_model_request(self, *, step: int, input_tokens: int, model: str = "scripted") -> Event:
        return await self.emit(EventType.MODEL_REQUEST, name=model, step=step)

    async def emit_model_response(
        self, *, step: int, input_tokens: int, output_tokens: int, preview: str,
        model: str = "scripted", token_source: str = "estimated",
    ) -> Event:
        """``token_source``: "estimated" (chars/4 heuristic) or "reported"
        (usage numbers returned by a real backend)."""
        return await self.emit(
            EventType.MODEL_RESPONSE, name=model, step=step, content_preview=preview[:200],
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            token_source=token_source,
        )

    async def finish(self, status: str) -> Trace:
        """Freeze and return the trace. Further emits are ignored."""
        if not self._finished:
            self._finished = True
            self.duration_ms = self.now_offset_ms()
        return Trace(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            agent_id=self.agent_id,
            status=status,
            started_at=self._t0_wall,
            duration_ms=round(self.duration_ms, 3),
            events=list(self.events),
        )

    duration_ms: float = 0.0
