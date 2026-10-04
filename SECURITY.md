# Security Policy

## Scope and stance

AgentReliabilityLab is a **test harness**. Its hard rules:

1. **No destructive interaction with real external systems — ever.** All
   high-risk scenarios run inside mocks/sandboxes:
   - Tool handlers are in-process functions. There is no HTTP client in the
     tool gateway; `http` tools are scripted status/body sequences, and `effect`
     tools (email/webhook/transfer/delete) append records to an **in-process
     outbox**.
   - Effect destinations are validated against the scenario's
     `env.allowlisted_domains` (default `api.company.example`), which contains
     only RFC-2606 reserved `.example` names.
2. **Fake secrets only.** Secrets referenced in scenarios are **synthetic
   canaries**: `arl-canary-<sha256(scenario_id:name)[:16]>`, derived at run
   time. No real credential is ever read from the environment, and none exists
   in the repository. If you find a canary value in a report, that is the
   leakage evaluator doing its job — the value is a deterministic hash, not a
   secret.
3. **Agents under test are sandboxed by construction.** An adapter can only
   act through `ToolEnvironment.call_tool`; there is no other capability exposed
   to the agent process. Prompt-injection content lives in mock document
   corpora and never executes.

## Reporting a vulnerability

Open a private security advisory via GitHub ("Security → Report a
vulnerability") or contact the maintainers directly. Please do not open public
issues for exploitable findings. Expected response: 7 days.

## Threat model (what ARL defends against)

ARL's threat model covers the **agent under test**, not the harness itself —
see [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for the full STRIDE-style
breakdown (prompt injection, exfiltration via tool arguments, unauthorized
actions, approval bypass, secret leakage, unsafe state reuse).

What the harness itself guarantees:

- Runs are hermetic: no network egress from the tool gateway, no DNS, no
  subprocesses.
- The dashboard binds to `127.0.0.1` by default and serves read-only data from
  the local SQLite file.
- Tests never require GPU, network, or credentials (CI enforces this).

## Data handling

- Traces, reports and the SQLite DB contain only scenario data and canary
  values.
- If you point the `openai:` adapter at a real backend, the *task text* you
  wrote is what gets sent — scenario content is your data. Nothing else leaves
  the process, and the offline reference agents never do.
