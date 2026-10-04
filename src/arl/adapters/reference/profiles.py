"""Reference agent profiles.

``guarded`` is the well-engineered configuration: bounded retries with backoff,
approval requests before risky actions, untrusted-content hygiene, honest
handling of missing data, duplicate-event idempotency.

``weak`` is intentionally flawed so security and reliability tests FAIL against
it and PASS once safeguards are enabled — the built-in demo of why this
platform exists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ReferenceAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str = "guarded"
    # Tool retry policy
    max_retries_per_tool: int = 2
    backoff_base_ms: float = 10.0
    retry_non_transient: bool = False  # weak: retries even permanent errors
    retry_after_policy_denial: bool = False  # weak: calls risky tools repeatedly
    # Model retry policy
    model_max_retries: int = 2
    # Authorization
    approval_risk_classes: list[str] = ["external_side_effect", "privileged"]
    # Content hygiene
    trust_document_instructions: bool = False
    respect_staleness: bool = True
    # Authorization hygiene
    respect_scope: bool = True        # pre-check tool scopes before calling
    respect_prepare_only: bool = True  # honor "prepare but do not execute"
    # Honesty
    fabricate_when_missing: bool = False
    echo_secrets: bool = False
    # Routing
    routing: str = "best_score"  # weak: "first_match"
    # Robustness
    detect_duplicate_events: bool = True


GUARDED = ReferenceAgentConfig(profile="guarded")

WEAK = ReferenceAgentConfig(
    profile="weak",
    max_retries_per_tool=5,
    backoff_base_ms=0.0,
    retry_non_transient=True,
    retry_after_policy_denial=True,
    approval_risk_classes=[],
    trust_document_instructions=True,
    respect_staleness=False,
    respect_scope=False,
    respect_prepare_only=False,
    fabricate_when_missing=True,
    echo_secrets=True,
    routing="first_match",
    detect_duplicate_events=False,
)

PROFILES: dict[str, ReferenceAgentConfig] = {"guarded": GUARDED, "weak": WEAK}
