"""Canary secrets. Security scenarios never use real credentials: every secret
is a synthesized, scenario-stable canary string. Leakage evaluators scan the
trace for these canaries, so any leak is provably a leak of a fake value."""

from __future__ import annotations

import hashlib

from arl.scenarios.schema import EnvSpec

CANARY_PREFIX = "arl-canary-"


def synthesize_canary(scenario_id: str, name: str) -> str:
    digest = hashlib.sha256(f"{scenario_id}:{name}".encode()).hexdigest()[:16]
    return f"{CANARY_PREFIX}{digest}"


def build_secret_map(env_spec: EnvSpec, scenario_id: str) -> dict[str, str]:
    """Named fake secrets for this scenario, e.g. ``{"API_KEY": "arl-canary-..."}``."""
    return {name: synthesize_canary(scenario_id, name) for name in env_spec.secrets}


def substitute_secret_refs(value: object, secrets: dict[str, str]) -> object:
    """Replace ``{{SECRET:NAME}}`` placeholders in strings with canary values,
    recursively through dicts/lists. Lets scenario authors plant secrets in
    documents or tool data without hard-coding values."""
    if isinstance(value, str):
        out = value
        for name, canary in secrets.items():
            out = out.replace("{{SECRET:" + name + "}}", canary)
        return out
    if isinstance(value, dict):
        return {k: substitute_secret_refs(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_secret_refs(v, secrets) for v in value]
    return value
