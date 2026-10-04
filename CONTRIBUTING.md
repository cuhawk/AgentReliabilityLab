# Contributing

## Setup

```bash
git clone <repo> && cd p3
uv sync                # Python 3.12; installs runtime + dev deps
uv run arl demo        # sanity check: offline weak-vs-guarded demo
```

## Gates (all must pass before merge)

```bash
uv run ruff check src tests examples      # lint (line length 120)
uv run mypy                               # strict-ish: disallow_untyped_defs
uv run python -m pytest tests -q          # platform self-tests
uv run arl eval suites/ci.yaml            # eval gate: guarded agent must pass
```

The same gates run in CI (`.github/workflows/ci.yml`), plus the demo smoke and
a Docker build.

Rules of the road:

- **CPU-only.** No test may require a GPU. The platform's determinism comes
  from mock tools and scripted faults, not model calls.
- **No network in tests.** Use `httpx.MockTransport` for anything HTTP.
- **No real secrets, ever.** Secrets in scenarios are synthetic canaries;
  see [SECURITY.md](SECURITY.md).
- **No destructive external interaction.** `effect` tools write to an
  in-process outbox; nothing dials out.

## What to work on

### Adding scenarios (highest value)

1. Copy the closest existing file in `scenarios/`.
2. Keep one behavior under test per scenario; expectations are what bind.
3. Validate: `uv run arl scenarios validate scenarios/<file>.yaml`.
4. Prove it can fail: `uv run arl eval scenarios/<file>.yaml --agent reference:weak --no-fail`
   (exit 1 = the expectations bite).
5. Add it to the right suite (`suites/full.yaml` at minimum).

### Adding evaluators

Subclass `arl.evaluation.base.Evaluator` (implement `evaluate(ctx) ->
list[CheckResult]`), append it to the engine's evaluator list, and add tests.
Rules: pure function of (scenario, trace, env snapshot, violations); never
call a model; return `not_applicable` (not `passed`) when the expectation
isn't set.

### Adding adapters

See [docs/ADAPTERS.md](docs/ADAPTERS.md). Test it offline with
`httpx.MockTransport`, mirroring `tests/test_adapters.py`.

### Adding faults

Add a `FaultType` member in `src/arl/scenarios/schema.py`, put it in the right
domain frozenset (tool/model/retrieval/state/run), implement its effect in the
gateway executor/handlers, and add a scenario + platform test. Cross-domain
fault schedules must fail at load time.

## Conventions

- Python 3.12 (StrEnum, modern typing: `X | None`, `list[...]`).
- Pydantic v2 models with `extra="forbid"` for anything user-authored.
- Line length 120 (reporting/dashboard template strings).
- Every bug fix ships with a regression test that failed before it.
- Doc decisions in `PLAN.md` (the build log) when they change semantics —
  e.g. the unauthorized-action semantics and span ownership are documented
  there.

## Running the dashboard during development

```bash
uv run arl eval suites/demo.yaml --db .arl/lab.db
uv run arl serve        # http://127.0.0.1:8600
```
