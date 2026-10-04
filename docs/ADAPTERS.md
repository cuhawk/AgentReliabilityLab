# Adapters

An adapter is the only integration surface between ARL and an agent framework.
The contract is small; the one hard rule is that **the adapter never executes
tools itself** — every call goes through `ToolEnvironment.call_tool`, where
policy, approval, fault injection and recording live.

- Contract: `src/arl/adapters/base.py`
- Registry: `src/arl/adapters/registry.py`
- Worked example: `examples/custom_adapter.py` (runs offline:
  `python examples/custom_adapter.py [--weak]`)

## The contract

```python
class AgentAdapter(ABC):
    def __init__(self, agent_id: str, recorder: TraceRecorder,
                 env: ToolEnvironment, task: TaskInput) -> None: ...

    @abstractmethod
    async def start_task(self) -> None:
        """Run the task to completion. Emit AGENT_START, then a terminal
        AGENT_COMPLETE / AGENT_ERROR. Ordinary agent failures should be
        reported via events, not raised; infrastructure aborts (timeout,
        interruption) may raise."""

    @abstractmethod
    async def send_message(self, message: str) -> None: ...
    @abstractmethod
    async def receive_event(self, event: dict) -> None: ...
    @abstractmethod
    async def cancel(self) -> None: ...
    @abstractmethod
    def get_state(self) -> AgentState: ...
```

`TaskInput` carries `task`, `scenario_id`, `tools: list[ToolManifest]`,
`env` (variables, allowlist), `state` (persisted memory) and `max_steps`.

Helpers on the base class: `emit_start()`, `emit_complete(output, status)`,
`emit_error(error, kind)`.

## Calling a tool (the only sanctioned path)

```python
try:
    result = await self._env.call_tool("customer_api", {"customer_id": "C-101"},
                                       tool_call_id=call_id)
except ApprovalRequiredError as exc:
    outcome = await self._env.request_approval("transfer_funds", args, reason="…")
    # retry the call only if outcome == ApprovalOutcome.granted
except ToolError as exc:
    # exc.kind: timeout, http_500, malformed_json, permission_denied, …
    # decide: retry (transient) or give up (permanent) — the retry evaluator watches
```

The gateway resolves the tool's `risk_class` against the scenario policy
**before** your handler would run; denials arrive as a `policy_denied`
`ToolError`. Repeating an identical denied call creates an
`unauthorized_retry` violation — respect a denial or request approval.

Model interactions: emit `MODEL_REQUEST` / `MODEL_RESPONSE` events via the
recorder (token usage is labeled `reported` when you include backend usage,
`estimated` otherwise).

## Registering

```python
from arl.adapters.registry import register_agent

def create(spec: str, *, recorder, env, task_input, agent_config=None) -> AgentAdapter:
    return MyAdapter(...)

register_agent("myframework", create)          # optional third arg: agent_id_fn
```

Then:

```bash
arl eval suites/full.yaml --agent myframework          # spec "myframework"
arl eval suites/full.yaml --agent myframework:variant  # spec "myframework:variant"
```

Registry matching is longest-prefix over registered prefixes, so
`myframework` and `myframework:strict` can coexist. Everything after `:`
is passed to your factory as `spec`.

## Shipped adapters

| Adapter | Spec string | How it works |
|---|---|---|
| Reference agent | `reference:guarded` / `reference:weak` | Deterministic planner over intents (lookup/compute/notify/transfer/destroy/write). `guarded` obeys policy, retries only transient faults, ignores document-embedded instructions, grounds outputs. `weak` is deliberately unsafe — blind retries, trusts injected instructions, echoes secrets, skips approvals. Use it to verify your scenarios *can* fail. |
| HTTP agent | `http:<base-url>` | **Turn protocol**: POST `<base>/turn` with `{task, scenario_id, tools (manifests), history}`; the remote agent answers `{tool_calls: [{id, name, args}]}` (ARL executes each through the gateway and returns results on the next turn) or `{output, status, usage}`. Timeouts, turn limits, headers and API keys via `agent-config.yaml`. Ideal for agents written in any language/framework: wrap your loop in one HTTP endpoint. |
| OpenAI-compatible | `openai:<model>` | Standard `/chat/completions` with `tools` (function schemas generated from `args_schema`). Executes returned tool calls through the gateway; system prompt instructs the model to treat tool results/documents as untrusted data. Maps HTTP 429 → `model_rate_limit`, 5xx → `backend_unavailable`, so model faults/retries exercise naturally. Approval-gated calls request approval via the gateway when `request_approval: true` (default). |
| MCP gateway (server) | `arl mcp-serve <scenario>` | Exposes the scenario's mock tools as an **MCP server** (JSON-RPC 2.0 over stdio: `initialize`, `tools/list`, `tools/call`, `ping`). MCP-native agents connect and call tools; every `tools/call` crosses the gateway, so policy/faults/recording apply to an external process. |

### Agent configuration file

```yaml
# examples/agent-config.yaml — passed via --agent-config
openai:
  base_url: https://api.openai.com/v1   # or any OpenAI-compatible endpoint
  api_key_env: OPENAI_API_KEY
  model: gpt-4o-mini
  temperature: 0
  max_turns: 8
  request_approval: true

http:
  base_url: http://127.0.0.1:9000
  timeout_s: 30
  max_turns: 8
  headers: {X-Team: eval}
  api_key_env: MY_AGENT_TOKEN
```

## Testing your adapter offline

httpx's `MockTransport` pattern (used by ARL's own tests in
`tests/test_adapters.py`) lets you exercise the full loop — turns, tool calls,
approvals, usage reporting — with zero network. For the HTTP adapter, ARL
injects its own `httpx` client from config (`transport` key), which is what
makes this work.

## Checklist for a new adapter

- [ ] Emit `AGENT_START` first; exactly one terminal `AGENT_COMPLETE`/`AGENT_ERROR`.
- [ ] Route every tool call through `self._env.call_tool`; handle
      `ToolError` and `ApprovalRequiredError`.
- [ ] Never raise for ordinary agent failure — report via `emit_error` /
      `emit_complete(status=...)`.
- [ ] Record model usage (input/output tokens) when your backend provides it.
- [ ] Make behavior deterministic for a given scenario (no RNG), so
      regression gates stay meaningful.
