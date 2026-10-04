# AgentReliabilityLab — Engineering Plan

Status: **living document** — updated at the end of every phase. Design decisions are
recorded here as they are made, with rationale, so contributors can disagree with
specifics instead of vibes.

## Goal

A framework-agnostic platform that executes controlled tasks against autonomous /
tool-using LLM agents, records the full execution trace, injects faults, and scores the
run with **deterministic evaluators first** (LLM judging is optional and always labeled).
The killer workflow is regression comparison of agent version A vs B in CI.

Pipeline:

```
Task → Agent → Tool calls → Fault Injection → Trace Recorder
    → Deterministic Evaluators → Security Evaluators → optional LLM Judge
    → Score / Regression Report
```

## Phase plan

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Scaffold, toolchain (uv, ruff, mypy, pytest), PLAN.md | ✅ |
| 1 | Event schema, Trace, Scenario DSL, Suite loader | ✅ |
| 2 | Trace recorder, fault injector, mock tool environment, policy engine, HITL gate | ✅ |
| 3 | Reference agent (weak/guarded profiles) + scripted deterministic model client | ✅ |
| 4 | Deterministic evaluators (13) + metric aggregation with CIs | ✅ |
| 5 | Runner, SQLite storage, cost model, `arl` CLI | ✅ |
| 6 | 60+ scenario YAMLs across 13 categories, suites (ci/full/security/demo) | ✅ |
| 7 | Regression engine: baseline vs candidate, thresholds, CI gate | ✅ |
| 8 | OpenTelemetry nested spans (`eval.run → agent.task → llm.call / tool.x / retry`) | ✅ |
| 9 | Adapters: generic HTTP, OpenAI-compatible tool-calling, MCP (client + ARL-as-MCP-gateway) | ✅ |
| 10 | Reporting: JSON / Markdown / self-contained HTML | ✅ |
| 11 | Dashboard: runs, metrics, regressions, per-scenario trace viewer | ✅ |
| 12 | Platform self-tests: evaluator correctness, metric math, parser, policy, compare gates | ✅ 53 tests, ruff + mypy clean, CPU-only |
| 13 | Docker, GitHub Actions, docs (README/ARCHITECTURE/EVALUATION/SECURITY/THREAT_MODEL/CONTRIBUTING/ADAPTERS/SCENARIO_DSL) | ✅ |

## Key design decisions

### D1 — ARL owns the tool environment (the "managed tool gateway")

The single most important architectural decision. Agent adapters do not execute tools
themselves; they call `env.call_tool(name, args)` on an environment ARL constructed from
the scenario. This gives ARL a chokepoint to (a) inject faults, (b) enforce
authorization policy, (c) record every call with arguments, and (d) keep all destructive
actions inside mocks. This holds for *every* shipped adapter, including the
OpenAI-compatible one: it executes returned tool calls through the gateway and appends
results as `role: "tool"` messages — there is no observational/dual path to keep honest.
Adapters that genuinely must run tools out of reach (an external HTTP agent executing
server-side) exercise policy/faults only when their remote loop routes back through the
gateway (the turn protocol does; an MCP client connecting to `arl mcp-serve` does).

Rationale: security testing through the same path as reliability testing. ARL also
*serves* the managed tools over the Model Context Protocol (`arl mcp-serve <scenario>`:
JSON-RPC 2.0 on stdio, `initialize`/`tools/list`/`tools/call`/`ping`), so any
MCP-capable agent can be evaluated without writing an adapter — every `tools/call`
crosses the gateway. (Direction chosen deliberately: ARL as MCP *server*, not as an
MCP client driving agent subprocesses — the client side adds a process-management
surface without adding a testing capability the HTTP/turn-protocol adapter lacks.)

### D2 — Events are the universal currency

Everything an evaluator or metric needs is derived from one typed event stream
(`EventType`), recorded with monotonic sequence numbers and offsets. Tool calls, retries,
approvals, policy decisions, injected faults, token usage: all events. Evaluators are
pure functions of `(scenario, trace, environment snapshot)`. This makes scoring
auditable — no hidden state, no "trust me" metrics.

Extra event types beyond the brief (documented additions): `POLICY_CHECK`,
`HUMAN_APPROVAL_RESULT`, `FAULT_INJECTED`, `EVIDENCE_UPDATE` (retrieval), `STATE_UPDATE`.
Rationale: the brief's event list has no place to record *outcomes* (approval decisions,
policy denials, injected faults); without them evaluators would have to guess.

### D3 — Deterministic by construction; wall-clock only where honest

The reference agent is a deterministic rule-driven loop over a scripted model client —
no network, no RNG in scoring paths. Fault schedules are keyed by call ordinal
(`call: 1 → timeout`, `call: 2 → success`), so a scenario always produces the same trace.
Latency metrics use real wall-clock (labeled as such), which is why latency budgets in
scenarios are generous and injected delays are tens of milliseconds, not seconds. Cost
and tokens are recorded per model call; for the reference agent they are clearly labeled
`source: estimated` (chars/4 heuristic); API adapters report `source: reported` usage.

### D4 — Deterministic evaluators first, LLM judge optional and labeled

Every `CheckResult` carries `kind: DETERMINISTIC | MODEL_JUDGED`. The semantic judge is
a separate optional module that is off by default, requires an explicit endpoint, and
its scores never blend silently into deterministic aggregates. Unsupported-claim
detection for the reference agent is fully deterministic: every value in the final
answer must be traceable to evidence recorded in the trace (tool results / documents).

### D5 — Risk classes + scope, enforced at the gateway

Tools carry `READ | WRITE | EXTERNAL_SIDE_EFFECT | PRIVILEGED`. The policy engine maps
risk class → allow / allow-in-scope / require-approval / deny, with optional per-tool
argument scopes (e.g. `account_id ∈ {C-101}`). Blocked calls still produce
`POLICY_CHECK: denied` + violation records, so "agent attempted an unauthorized call" is
measurable even though the gateway refused execution.

### D6 — SQLite first, storage as an interface

`Storage` is an ABC; SQLite is the shipped implementation (WAL, JSON columns). Postgres
is an extension point, not a second implementation maintained by hand — dual-dialect SQL
in a v0 is how bugs are born. The eval engine never requires the dashboard; the
dashboard is a FastAPI app that only reads storage.

### D7 — Regression gates are threshold-based, statistics are advisory

CI gates compare baseline vs candidate per metric with configurable absolute/relative
thresholds and a minimum-effect floor (noise immunity). Wilson 95% intervals are
computed and reported; when baseline and candidate intervals for a rate metric do not
overlap, the comparison row is flagged `statistically_significant` — but gates fire on
configured thresholds, not on CI overlap, because flaky CI is worse than a conservative
gate. This is documented in `EVALUATION.md`.

### D8 — Safety: everything destructive is a mock

No scenario touches a real external system. "External" tools (`email`, `webhook`,
`transfer`, `delete`) write to an in-process outbox; the only "network" is loopback in
tests. Secrets in security scenarios are synthesized canary strings
(`arl-canary-…`), never read from the real environment. The threat model
(`docs/THREAT_MODEL.md`) covers the platform's own attack surface too (evaluator
manipulation, poisoned scenarios).

### D9 — Authorization semantics: a respected denial is the gateway working

Made explicit during evaluator hardening (Phase 12). A *first* policy denial that the
agent respects is correct system behavior — recorded as a `POLICY_CHECK: denied` event,
not a violation. Violations are: `unauthorized_retry` (repeating an identical denied
call, or attempting an approval-gated call without ever requesting approval) and
`unknown_tool` (calling a tool that does not exist). The `authorization` evaluator fails
scenarios on violations; `expect.no_unauthorized_actions: true` opts into strict mode
where *any* denial fails (high-assurance scenarios). Rationale: conflating enforcement
with misbehavior made every "policy blocks the bad thing" scenario unpassable, which
would have trained users to write permissive policies.

### D10 — Span ownership: the EventSpanTracker owns `agent.task`

The runner does not open a span around the agent call. `EventSpanTracker` (an event
subscriber) opens `agent.task` on `AGENT_START`, closes it on `AGENT_COMPLETE/ERROR`,
and parents *all* child spans (tool/llm/retry/policy/approval/fault) explicitly via
`set_span_in_context`. Rationale: two competing `agent.task` spans (runner + tracker)
broke nesting tests, and ambient-context parenting silently reparented child spans to
the wrong ancestor under async.

### D11 — `ruff check` is the lint gate; `ruff format` is not enforced

`ruff check` (E,F,I,W,UP,B,SIM,ASYNC,RUF; line length 120) runs clean over
`src`/`tests`/`examples`. Enforcing the formatter project-wide at this stage would
reformat every file for zero behavioral gain; the CI runs the checker, not the
formatter.

## Verification log (final state)

- `uv run python -m pytest tests -q` → **53 passed** (~1.7 s; CPU-only — no GPU anywhere,
  per constraint).
- `uv run ruff check src tests examples` → **clean** (after StrEnum migration of all
  enums to `enum.StrEnum`, Python 3.12).
- `uv run mypy` → **clean** (51 source files; `disallow_untyped_defs`; dashboard excluded).
- `arl eval suites/full.yaml --agent reference:guarded` → **62/62 pass**.
- Same suite with `reference:weak` → security scenarios fail with deterministic
  evidence (unauthorized retries, canary leakage, non-allowlisted effect targets,
  approval bypasses) — the demo/regression contrast.
- `arl demo` → weak vs guarded runs + threshold comparison + verdict, fully offline.
- `arl serve` → dashboard over the SQLite DB (runs, metric CIs, per-scenario checks,
  violations, tool calls, trace).
- Docker/CI wiring (`Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`) is
  exercised by the CI `docker` job; no Docker daemon available on the dev machine that
  produced this repo.

## Non-goals (v0)

- Distributed runs (Redis/queue) — interfaces leave room, no implementation yet.
- Postgres storage — interface defined, SQLite shipped.
- A JS build step for the dashboard — server-rendered + vanilla JS, zero build tooling.
- Training/fine-tuning anything. This is a measurement platform.
