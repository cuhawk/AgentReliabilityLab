# AgentReliabilityLab (ARL)

Framework-agnostic testing platform for **autonomous, tool-using LLM agents** —
reliability, correctness, recoverability, security, safety boundaries, cost and
latency.

ARL is **not** an LLM-as-judge wrapper. Evaluation is deterministic and
reproducible: scenarios are declarative YAML, tools are sandboxed mocks, faults
are injected on exact call ordinals, and every check is a pure function of the
recorded trace. Metrics come with Wilson confidence intervals and are never
fabricated — if a value cannot be measured, it is reported as missing.

```
Task → Agent adapter → Tool calls → Fault injection → Trace recorder
     → Deterministic evaluators → Metrics (95% CIs) → Regression gates → Reports / dashboard
```

- Python 3.12+ · Pydantic v2 · asyncio · FastAPI · SQLite · OpenTelemetry · zero API keys required

---

## What is tested

**62 scenarios across 13 categories** ship with the repo (all offline, all
deterministic):

| Category | Scenarios | What it probes |
|---|---|---|
| `happy_path` | 8 | task completion, output contracts, multi-tool composition |
| `tool_failure` | 9 | timeouts, 429/500s, malformed JSON, empty/partial results, retry discipline |
| `model_failure` | 4 | rate limits, backend outages, truncated/invalid structured output |
| `routing` | 5 | right tool, right entity; no tool when none is needed |
| `retrieval_failure` | 4 | stale, irrelevant, conflicting or missing documents |
| `state_recovery` | 4 | missing/stale state, duplicate events (idempotency), interruption |
| `authorization` | 5 | READ/WRITE/EXTERNAL_SIDE_EFFECT/PRIVILEGED policy, scope limits |
| `human_approval` | 6 | approval ordering, deny-and-stop, timeouts, prepare-only mode |
| `prompt_injection` | 7 | direct & indirect injection, multi-step attacks, constraint overrides |
| `secret_handling` | 3 | canary secrets never leak into output, state or tool args |
| `hallucination` | 3 | no phantom citations, no invented fields, no unsupported claims |
| `cost` | 3 | token/cost budgets, bounded retry spend |
| `latency` | 1 | per-task latency budgets |

Each scenario pairs a **task** with a **mock tool environment** (data, compute,
retrieve, effect and http tools), an **authorization policy**, a **simulated
human** for approvals, and a set of deterministic **expectations**.

## How faults are injected

There are no random seeds. Faults are **ordinal-keyed**: "the 2nd call to
`customer_api` gets HTTP 500" (`faults: [{call: 2, fault: http_500}]`) or
"every 3rd call times out" (`{every: 3, fault: timeout}`). Runs replay
identically, so a failure is always explainable and a fix is always provable.

**21 fault types** across five domains:

| Domain | Faults |
|---|---|
| Tool (8) | `timeout`, `http_429`, `http_500`, `malformed_json`, `empty_result`, `delay`, `partial_result`, `permission_denied` |
| Model (4) | `model_rate_limit`, `backend_unavailable`, `truncated_response`, `invalid_structured_output` |
| Retrieval (5) | `irrelevant_doc`, `stale_doc`, `conflicting_evidence`, `missing_source`, `malicious_doc` |
| State (2) | `missing_state`, `stale_state` |
| Run (2) | `duplicate_event` (replays the previous tool result — idempotency), `interrupted` (aborts mid-run) |

All tool calls are routed through a **managed gateway** (`ToolEnvironment.call_tool`):
policy check → fault injection → sandboxed handler → trace event. Agents never
touch the network; `effect` tools write to an in-process outbox and are checked
against a domain allowlist. Secrets are **synthetic canaries** — deterministic
per-scenario values (`arl-canary-<hash>`) that never exist outside the sandbox.

## Which metrics are deterministic

**13 evaluators**, each a pure function of (scenario, trace, environment
snapshot, policy violations): `completion`, `output_format`, `tools`,
`tool_args`, `retry`, `state`, `evidence`, `authorization`, `approval`,
`secrets`, `injection`, `cost`, `latency`.

Aggregate metrics carry **Wilson 95% confidence intervals**; latency is
reported as p50/p95 with percentile intervals. Nothing is averaged away
silently: a metric that cannot be computed is omitted, and token counts are
labeled `estimated` (chars/4) unless the backend reports real usage.

Every scenario check is one of `passed` / `failed` / `not_applicable` with the
observed evidence attached (see [EVALUATION.md](EVALUATION.md)).

## How to compare two agents

```bash
arl eval suites/full.yaml --agent reference:guarded --output-dir .arl/guarded --db .arl/lab.db
arl eval suites/full.yaml --agent reference:weak    --output-dir .arl/weak    --db .arl/lab.db

arl compare .arl/guarded/report.json .arl/weak/report.json --fail-on-regression
```

The comparison applies explicit, configurable **regression gates** (defaults
below; overrides via a thresholds YAML):

| Metric | Gate |
|---|---|
| `task_success_rate` | drop ≤ 3% (deltas ≤ 0.5% treated as noise) |
| `tool_selection_accuracy` | drop ≤ 3% |
| `tool_argument_accuracy` | drop ≤ 5% |
| `recovery_success_rate` | drop ≤ 5% |
| `unauthorized_action_rate` | hard cap 0.0 on the candidate |
| `injection_resistance_rate` | drop ≤ 0.0 (any real drop fails) |
| `approval_compliance_rate` | drop ≤ 0.0 |
| `p95_latency_ms` | increase ≤ 25% (noise floor 50 ms) |
| `cost_per_successful_task_usd` | increase ≤ 20% |

Wilson-interval overlap is reported as statistical evidence but **never**
silently decides a gate — the gate list does. Exit code 1 on regression, so it
works as a CI step. See [EVALUATION.md](EVALUATION.md#regression-testing).

## How to run in CI

```yaml
- run: pip install agent-reliability-lab        # or: uv sync
- run: arl eval suites/ci.yaml                  # exits 1 if the agent misses expectations
```

`suites/ci.yaml` is a 12-scenario deterministic gate (task success, routing,
retry recovery, authz, injection resistance, secret handling) that passes for
the guarded reference agent with no network and no API keys. The bundled
[.github/workflows/ci.yml](.github/workflows/ci.yml) runs ruff + mypy + pytest,
the eval gate, the offline demo, and a Docker-image smoke test on every PR.

`arl compare` is the second CI primitive: gate a candidate agent against a
stored baseline report before merge.

## Quickstart

```bash
git clone <repo> && cd p3            # repo root
uv sync                              # installs arl + dev tools

arl demo                             # weak vs guarded agent on the demo suite,
                                     # + regression comparison. Fully offline.
arl eval suites/demo.yaml            # one suite, guarded agent
arl eval scenarios/fail-timeout-retry-success.yaml --agent reference:weak --no-fail
arl serve                            # dashboard at http://127.0.0.1:8600
```

Docker:

```bash
docker build -t arl .
docker run --rm arl demo             # one-command offline demo
docker compose up dashboard          # dashboard on :8600 (after compose run demo)
```

## Agents under test

An **adapter** is the only integration surface (see
[docs/ADAPTERS.md](docs/ADAPTERS.md)): `start_task` / `send_message` /
`receive_event` / `cancel` / `get_state`, plus a typed event stream. Every tool
call must go through the environment gateway — that is where faults, policy,
approvals and recording happen.

Shipped adapters:

| Adapter | Spec | Notes |
|---|---|---|
| `reference:guarded` / `reference:weak` | built-in | strong vs deliberately unsafe behavior profiles; the weak agent is the demo/regression contrast |
| `http:<base-url>` | HTTP agent | turn protocol: POST `/turn` with tool manifests → tool calls or final output |
| `openai:<model>` | OpenAI-compatible | any `/chat/completions` endpoint with tool calling |
| MCP | `arl mcp-serve <scenario>` | exposes scenario tools over MCP stdio; every `tools/call` crosses the gateway |

Custom frameworks register in three lines:

```python
from arl.adapters.registry import register_agent
register_agent("myframework", my_factory)
# then: arl eval suites/full.yaml --agent myframework:variant
```

## Dashboard

`arl serve --db .arl/lab.db` serves a dependency-free UI (runs list, metric
cards with CIs, per-scenario checks/violations/tool calls, rendered trace).
The dashboard reads SQLite only — the eval engine never depends on it.

## Repository layout

```
src/arl/
  adapters/        agent adapters + registry + MCP gateway
  core/            events, ids
  environment/     managed tool gateway: handlers, policy, faults wiring, secrets
  evaluation/      13 deterministic evaluators, engine, aggregates (Wilson CIs)
  faults/          ordinal-keyed fault injector
  observability/   OpenTelemetry span tracking (eval.run → agent.task → tool.x)
  recording/       trace recorder + event model + SQLite storage
  regression/      thresholds, A/B comparison, gate decisions
  reporting/       JSON / Markdown / HTML report renderers
  runner.py        orchestration: build env, run agent, evaluate, store, report
  scenarios/       scenario DSL schema + loader (+ shared _library.yaml)
  dashboard/       FastAPI app + static SPA
scenarios/         62 scenario YAMLs
suites/            ci / demo / security / full
tests/             53 platform self-tests (CPU-only)
docs/              THREAT_MODEL, ADAPTERS, SCENARIO_DSL
examples/          custom adapter, agent config, MCP demo, budget gating
```

## Security stance

- **Everything destructive stays inside mocks/sandboxes.** `effect` tools
  append to an in-process outbox; nothing ever touches a real external system.
- **Fake secrets only.** Canary secrets are synthesized per scenario; no real
  credentials exist in the repo or the test paths.
- Threat model: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) ·
  security policy: [SECURITY.md](SECURITY.md).

## Development

```bash
uv sync
uv run python -m pytest tests -q     # 53 tests, ~2 s, CPU-only (no GPU)
uv run ruff check src tests examples
uv run mypy
```

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) · architecture deep-dive:
[ARCHITECTURE.md](ARCHITECTURE.md).

## License

MIT — see [LICENSE](LICENSE).
