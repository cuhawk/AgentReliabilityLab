"""Generic HTTP agent adapter.

Drives a remote agent that speaks the ARL *turn protocol* (docs/ADAPTERS.md):
each turn the adapter POSTs the task, the tool manifests and the tool results
gathered so far; the agent replies with either tool calls or a final answer.
The adapter executes every tool call locally through the injected
:class:`ToolEnvironment`, so fault injection, authorization policy, recording
and HITL simulation stay inside ARL regardless of where the agent runs.

Wire format (JSON):

    request  {"task": str, "scenario_id": str, "tools": [manifest...],
              "history": [ {"tool_call": {...}} | {"tool_result": {...}} ... ]}
    response {"tool_calls": [{"id": str, "name": str, "args": {...}}, ...]}
           | {"output": any, "status": "success"|"failed"|"blocked",
              "usage": {"input_tokens": int, "output_tokens": int}}

All requests go to ``POST {base_url}/turn``. Transport errors surface as
AGENT_ERROR. This adapter never touches real external systems beyond the
configured agent endpoint."""

from __future__ import annotations

import json
from typing import Any

import httpx

from arl.adapters.base import AgentAdapter, AgentState, AgentStatus
from arl.environment.errors import ApprovalRequiredError, ToolError


class HttpAgent(AgentAdapter):
    """Turn-protocol client for an HTTP-hosted agent."""

    def __init__(self, agent_id: str, recorder: Any, env: Any, task_input: Any,
                 *, base_url: str, config: dict[str, Any]) -> None:
        super().__init__(agent_id, recorder, env, task_input)
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(config.get("timeout_s", 30.0))
        self._max_turns = int(config.get("max_turns", task_input.max_steps))
        headers = dict(config.get("headers", {}))
        api_key_env = config.get("api_key_env")
        if api_key_env:
            import os

            key = os.environ.get(api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout_s,
                                         headers=headers, transport=_maybe_transport(config))
        self._state = AgentState(status=AgentStatus.idle)

    async def start_task(self) -> None:
        await self.emit_start()
        self._state.status = AgentStatus.running
        history: list[dict[str, Any]] = []
        status = "success"
        output: Any = None
        try:
            for turn in range(self._max_turns):
                if self._cancelled:
                    status = "failed"
                    output = "cancelled by platform"
                    break
                response = await self._turn(history)
                usage = response.get("usage") or {}
                tool_calls = response.get("tool_calls") or []
                if tool_calls:
                    preview = "tool calls: " + ", ".join(str(c.get("name")) for c in tool_calls)
                else:
                    output = response.get("output")
                    status = str(response.get("status", "success"))
                    preview = str(output)[:200]
                await self._recorder.emit_model_response(
                    step=turn, input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)), preview=preview,
                )
                if not tool_calls:
                    break
                for call in tool_calls:
                    result = await self._execute(call)
                    history.append({"tool_call": call})
                    history.append({"tool_result": result})
        except Exception as exc:
            await self.emit_error(f"{type(exc).__name__}: {exc}", kind="http_agent_error")
            self._state.status = AgentStatus.failed
            return
        await self.emit_complete(output, status=status)
        self._state.status = AgentStatus.completed

    async def _turn(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "task": self._task.task,
            "scenario_id": self._task.scenario_id,
            "tools": [m.model_dump() for m in self._task.tools],
            "history": history,
        }
        await self._recorder.emit_model_request(step=len(history), input_tokens=len(json.dumps(payload)) // 4)
        response = await self._client.post("/turn", json=payload)
        response.raise_for_status()
        return response.json()

    async def _execute(self, call: dict[str, Any]) -> dict[str, Any]:
        name = str(call.get("name", ""))
        args = dict(call.get("args", {}))
        manifest = next((m for m in self._task.tools if m.name == name), None)
        try:
            result = await self._env.call_tool(name, args, tool_call_id=str(call.get("id", name)))
            return {"call_id": call.get("id"), "ok": True, "result": result}
        except ApprovalRequiredError:
            if manifest is not None and manifest.risk_class.value in ("external_side_effect", "privileged"):
                outcome = await self._env.request_approval(name, args, reason="http agent requested execution")
                if outcome.decision.value == "approve":
                    result = await self._env.call_tool(name, args, tool_call_id=str(call.get("id", name)))
                    return {"call_id": call.get("id"), "ok": True, "result": result}
                return {"call_id": call.get("id"), "ok": False,
                        "error": f"human {outcome.decision.value} the request"}
            return {"call_id": call.get("id"), "ok": False, "error": "tool requires human approval"}
        except ToolError as exc:
            return {"call_id": call.get("id"), "ok": False, "error": f"{exc.kind}: {exc}"}

    async def send_message(self, message: str) -> None:
        self._state.memory.setdefault("messages", []).append(message)

    async def receive_event(self, event: dict[str, Any]) -> None:
        self._state.memory.setdefault("platform_events", []).append(event)

    async def cancel(self) -> None:
        self._cancelled = True

    def get_state(self) -> AgentState:
        return self._state


def _maybe_transport(config: dict[str, Any]) -> httpx.AsyncBaseTransport | None:
    """Tests may inject ``transport`` (e.g. httpx.MockTransport) via config."""
    transport = config.get("transport")
    return transport if isinstance(transport, httpx.AsyncBaseTransport) else None
