"""Agent adapter contract.

Adapters normalize any agent (reference, HTTP, OpenAI-compatible, MCP) into the
ARL event stream. The adapter does NOT execute tools: it resolves every tool
call through the injected :class:`ToolEnvironment`, which is where fault
injection, authorization policy and recording live (see PLAN.md, D1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from arl.core.events import EventType
from arl.environment.env import ToolEnvironment, ToolManifest
from arl.recording.recorder import TraceRecorder


class AgentStatus(StrEnum):
    idle = "idle"
    running = "running"
    waiting_approval = "waiting_approval"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AgentStatus = AgentStatus.idle
    steps: int = 0
    memory: dict[str, Any] = Field(default_factory=dict)


class TaskInput(BaseModel):
    """Everything the agent is given for one scenario run."""

    model_config = ConfigDict(extra="forbid")

    task: str
    scenario_id: str
    tools: list[ToolManifest]
    env: dict[str, Any] = Field(default_factory=dict)  # variables, allowlisted_domains
    state: dict[str, Any] = Field(default_factory=dict)  # agent's persisted memory
    max_steps: int = 12


class AgentAdapter(ABC):
    """Lifecycle: ``start_task`` runs the agent to completion (or error).

    ``send_message`` delivers user turns; ``receive_event`` delivers platform
    events (e.g. replayed tool results under the duplicate_event fault);
    ``cancel`` requests cooperative cancellation; ``get_state`` reports the
    current agent state.
    """

    def __init__(self, agent_id: str, recorder: TraceRecorder, env: ToolEnvironment, task: TaskInput) -> None:
        self.agent_id = agent_id
        self._recorder = recorder
        self._env = env
        self._task = task
        self._cancelled = False

    @abstractmethod
    async def start_task(self) -> None:
        """Execute the task to completion, emitting AGENT_START and a terminal
        AGENT_COMPLETE / AGENT_ERROR event. Must not raise for ordinary agent
        failures; infrastructure aborts (timeout, interruption) may raise."""

    @abstractmethod
    async def send_message(self, message: str) -> None:
        """Deliver a user message to a running or completed agent."""

    @abstractmethod
    async def receive_event(self, event: dict[str, Any]) -> None:
        """Deliver a platform event (e.g. duplicated tool result)."""

    @abstractmethod
    async def cancel(self) -> None:
        """Request cooperative cancellation."""

    @abstractmethod
    def get_state(self) -> AgentState:
        """Current agent state snapshot."""

    # --- shared helpers ---------------------------------------------------

    async def emit_start(self) -> None:
        await self._recorder.emit(EventType.AGENT_START, task=self._task.task)

    async def emit_complete(self, output: Any, status: str = "success", note: str = "") -> None:
        await self._recorder.emit(
            EventType.AGENT_COMPLETE, output=output, status=status, note=note,
        )

    async def emit_error(self, error: str, kind: str = "agent_error") -> None:
        await self._recorder.emit(EventType.AGENT_ERROR, error=error, kind=kind)


class ModelCallError(Exception):
    """Raised by model clients when the backend is unavailable / rate limited."""

    def __init__(self, message: str, kind: str, retriable: bool = True) -> None:
        super().__init__(message)
        self.kind = kind
        self.retriable = retriable


TaskStatus = Literal["success", "failed", "blocked"]
