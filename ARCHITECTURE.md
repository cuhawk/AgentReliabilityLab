# Architecture

This document explains how ARL works internally: the data model, the execution
pipeline, and the design decisions that keep evaluation deterministic and
honest.

## The core loop

```
runner.run_suite(suite_id, scenarios, config)
 │
 ├─ for each scenario:
 │   ├─ ToolEnvironment(scenario, recorder, injector)     # managed gateway
 │   ├─ FaultInjector(scenario, recorder)                 # ordinal-keyed schedules
 │   ├─ adapter = registry.create(agent_spec, ...)        # AgentAdapter
 │   ├─ adapter.start_task()                              # timeout-guarded
 │   │    └─ agent loop: plan → call_tool → observe → …
 │   │         └─ env.call_tool(name, args)
 │   │              ├─ policy engine decision            (allow / in-scope / approval / deny)
 │   │              ├─ approval round-trip (simulated human)
 │   │              ├─ fault injection                    (tool / run faults)
 │   │              ├─ sandboxed handler execution        (data/compute/retrieve/effect/http)
 │   │              └─ TOOL_REQUEST / TOOL_RESULT events
 │   ├─ trace = recorder.finish(status)                   # frozen event stream
 │   └─ evaluate_scenario(scenario, trace, env_snapshot, violations)
 │
 └─ aggregate_suite(results) → metrics with Wilson CIs → reports → storage
```

Three invariants hold throughout:

1. **Adapters never execute tools.** All tool access goes through
   `ToolEnvironment.call_tool`. There is no path from an adapter to a handler
   that bypasses policy, faults, and recording.
2. **Events are the universal currency.** Every observable action becomes a
   typed event (`AGENT_START`, `TOOL_REQUEST`, `TOOL_RESULT`, `MODEL_REQUEST`,
   `MODEL_RESPONSE`, `RETRY`, `POLICY_CHECK`, `HUMAN_APPROVAL_REQUEST/RESULT`,
   `EVIDENCE_UPDATE`, `FAULT_INJECTED`, `AGENT_COMPLETE/ERROR`, …) with a
   monotonic sequence number, `t_offset_ms` (via `time.perf_counter()` —
   `monotonic()` ticks ~15.6 ms on Windows, which collapses short durations)
   and a JSON payload.
3. **Evaluators are pure.** `evaluate()` receives `(scenario, trace, env
   snapshot, violations)` and returns checks. No I/O, no clock, no RNG — the
   same trace always yields the same verdict.

## Determinism model

- **Fault schedules are ordinal-keyed**, not stochastic: `{call: N}` /
  `{every: N}` against per-tool call counters, the model-call counter,
  state-read ordinals, or the global tool-call ordinal (run faults). The
  injector emits `FAULT_INJECTED` events so the trace shows *why* a call
  failed.
- **Canary secrets are derived**, not generated: `arl-canary-<sha256(scenario_id:name)[:16]>`.
  A secret in a report proves leakage without shipping anything real.
- **Token counts are labeled**: `estimated` (chars ÷ 4) unless an adapter
  reports backend usage (`token_source: "reported"`). Metrics never invent
  numbers — unpriced models produce `cost 0.0` with `source: unpriced`, not a
  guess.
- **Timing** uses `perf_counter()` deltas recorded at event time; latency
  budgets (e.g. 100 ms) bind even for millisecond-scale mock handlers.

## The managed tool gateway

`ToolEnvironment.call_tool(name, args, tool_call_id, attempt)`:

1. **Policy engine** — resolves the tool's `risk_class`
   (READ / WRITE / EXTERNAL_SIDE_EFFECT / PRIVILEGED) to an action
   (allow / allow_in_scope / require_approval / deny), checks argument scope
   (`scope: {field, allowed}`), emits `POLICY_CHECK`. A *first* denial is the
   gateway doing its job — recorded as `policy_denied`, not a violation.
   Repeating an identical denied call (or calling an approval-gated tool
   without ever requesting approval) raises an `unauthorized_retry` violation;
   calling an undefined tool raises `unknown_tool`.
2. **Simulated human** — `require_approval` parks the call in an approval
   queue; `env.request_approval` resolves against `HumanSpec` (approve/deny/
   timeout, delay), emitting request/result events and an `approval` OTel span.
3. **Fault injection** — the injector decides whether this ordinal is faulted
   and hands the fault to the executor.
4. **Handler** — the sandboxed implementation for the tool kind. `effect`
   handlers append an `EffectRecord` to the in-process **outbox**; destinations
   are checked against `env.allowlisted_domains`. Nothing dials out.
5. **Recording** — `TOOL_RESULT` with ok/error, fault, duration; retrieval
   results additionally emit `EVIDENCE_UPDATE` (the grounding evaluator reads
   these).

## Evaluation pipeline

`EvalContext` = scenario + trace + `{final_state, outbox, served_documents}` +
`violations`. The engine runs 13 evaluators (see [EVALUATION.md](EVALUATION.md)),
each returning `CheckResult` items (`passed`/`failed`/`not_applicable`,
expected vs observed, message). Checks with the same metric name aggregate into
`ScenarioResult.metrics`; `aggregate_suite` rolls scenario results into run
metrics with **Wilson 95% CIs** for rates and percentile intervals for p50/p95.

Scenario status = `pass` only if all applicable checks pass. Platform-level
failures (timeout, agent crash) override expectations — a scenario whose agent
timed out cannot "pass" by having set a lenient `final_status`.

## Regression testing

`compare_metrics(baseline, candidate, thresholds)` walks the gate table:
absolute drops (`max_drop`), relative drops (`max_drop_pct`), caps
(`candidate_max`), noise floors (`min_effect`), and direction-scoped gates (an
increase on a drop-gated metric is an improvement). Wilson-interval overlap is
displayed as evidence but gates are decided by the explicit thresholds only.
`compare_scenarios` adds a per-scenario flip table. The overall verdict drives
the CLI exit code. See [EVALUATION.md](EVALUATION.md#regression-testing).

## Observability

OpenTelemetry spans mirror the event stream:

```
eval.run
└── eval.scenario <id>
    ├── agent.task                      (opened on AGENT_START, closed on COMPLETE/ERROR)
    │   ├── llm.call                    (+ usage attributes when reported)
    │   ├── tool.<name>                 (status, fault, error_kind)
    │   ├── retry / policy.check / approval / fault.<type>
    └── evaluator spans                 (one per evaluator)
```

The `EventSpanTracker` subscribes to the recorder and parents every span to the
`agent.task` span explicitly (`set_span_in_context`), so nested spans survive
async context loss. Exporters: `none` (default), `console`, `otlp` (optional
extra).

## Storage & reporting

- `TraceRecorder` holds the event list; `finish()` returns the immutable trace.
- `SqliteStorage` persists runs, results, checks, metrics, violations, tool
  calls, and comparisons (thread-safe: `check_same_thread=False` + lock, since
  the dashboard serves from worker threads).
- `reporting.render` emits JSON (machine), Markdown (PR comments), and HTML
  (self-contained, with failed-scenario traces embedded).

## Dashboard

FastAPI app (`arl.dashboard.app.create_app`) over the SQLite file only:
`/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/scenarios/{sid}`,
`/api/comparisons`, plus a build-free static SPA. It has no import path into
the eval engine — delete it and `arl eval` is unaffected.

## Extending

- **Adapters**: implement `AgentAdapter`, register with
  `registry.register_agent(prefix, factory, id_fn)` — longest-prefix match on
  `--agent <prefix>:<variant>`. See [docs/ADAPTERS.md](docs/ADAPTERS.md).
- **Evaluators**: subclass `arl.evaluation.base.Evaluator`, append to the
  engine's evaluator list. Keep it pure; label any model-judged signal
  explicitly (`source: model_judged`) — deterministic evaluators never call a
  model.
- **Tool handlers**: `environment.handlers.HANDLERS` maps `compute` function
  names to pure async callables; scenario authors pick from this registry.
- **Faults**: add a `FaultType` member + handler mapping; keep it in the right
  domain set so cross-domain schedules are rejected at load time.
