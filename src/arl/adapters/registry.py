"""Agent registry: turn an agent spec string into a ready adapter.

Spec grammar:
  reference[:guarded|weak]      built-in deterministic reference agent
  openai:<model>                OpenAI-compatible tool-calling agent
                                (config via --agent-config YAML: base_url,
                                api_key_env, temperature, ...)
  http:<base-url>               generic HTTP agent speaking the ARL turn
                                protocol (docs/ADAPTERS.md)

Custom agents register a spec prefix via :func:`register_agent` (see
examples/custom_adapter.py).

MCP-native agents are evaluated the other way around: ARL serves the
scenario's tools as an MCP server (``arl mcp-serve <scenario>``) and the agent
connects to it, so every call crosses the ARL gateway (faults, policy,
recording) — see arl/adapters/mcp_gateway.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from arl.adapters.base import AgentAdapter, TaskInput
from arl.environment.env import ToolEnvironment
from arl.recording.recorder import TraceRecorder

AdapterFactory = Callable[..., AgentAdapter]
CustomIdFn = Callable[[str], str]

_CUSTOM: dict[str, AdapterFactory] = {}
_CUSTOM_ID: dict[str, CustomIdFn] = {}


def register_agent(prefix: str, factory: AdapterFactory, id_fn: CustomIdFn | None = None) -> None:
    """Register a custom adapter factory for specs starting with ``prefix``
    (matched longest-first, so ``example:naive`` beats ``example``)."""
    _CUSTOM[prefix] = factory
    if id_fn is not None:
        _CUSTOM_ID[prefix] = id_fn


def _match_custom(spec: str) -> tuple[str, AdapterFactory] | None:
    for prefix in sorted(_CUSTOM, key=len, reverse=True):
        if spec == prefix or spec.startswith(prefix + ":"):
            return prefix, _CUSTOM[prefix]
    return None


def agent_id_of(spec: str) -> str:
    """Stable agent identity for reports/storage."""
    matched = _match_custom(spec)
    if matched is not None and matched[0] in _CUSTOM_ID:
        return _CUSTOM_ID[matched[0]](spec)
    if spec == "reference" or spec.startswith("reference:"):
        profile = spec.split(":", 1)[1] if ":" in spec else "guarded"
        return f"reference-agent:{profile}"
    return spec


def create_adapter(
    spec: str,
    *,
    recorder: TraceRecorder,
    env: ToolEnvironment,
    task_input: TaskInput,
    agent_config: dict[str, Any] | None = None,
) -> AgentAdapter:
    agent_config = agent_config or {}
    matched = _match_custom(spec)
    if matched is not None:
        return matched[1](
            spec, recorder=recorder, env=env, task_input=task_input, agent_config=agent_config,
        )
    if spec == "reference" or spec.startswith("reference:"):
        from arl.adapters.reference.agent import ReferenceAgent
        from arl.adapters.reference.profiles import PROFILES

        profile = spec.split(":", 1)[1] if ":" in spec else "guarded"
        if profile not in PROFILES:
            raise ValueError(f"unknown reference profile '{profile}' (known: {', '.join(PROFILES)})")
        return ReferenceAgent(
            agent_id=f"reference-agent:{profile}", recorder=recorder, env=env,
            task_input=task_input, config=PROFILES[profile],
        )
    if spec.startswith("openai:"):
        from arl.adapters.openai_agent import OpenAICompatAgent

        model = spec.split(":", 1)[1]
        return OpenAICompatAgent(
            agent_id=f"openai:{model}", recorder=recorder, env=env, task_input=task_input,
            model=model, config=agent_config,
        )
    if spec.startswith("http:"):
        from arl.adapters.http_agent import HttpAgent

        base_url = spec.split(":", 1)[1]
        return HttpAgent(
            agent_id=f"http:{base_url}", recorder=recorder, env=env, task_input=task_input,
            base_url=f"http://{base_url}" if "://" not in base_url else base_url, config=agent_config,
        )
    raise ValueError(f"unknown agent spec '{spec}'")
