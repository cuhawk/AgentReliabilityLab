"""Token accounting and cost estimation.

Honesty rules:
  * The reference agent's "model" is a scripted client, so its token counts are
    a chars/4 *estimate* — recorded with ``token_source: estimated``.
  * API-backed adapters (OpenAI-compatible) record the backend's reported
    usage with ``token_source: reported``.
  * Prices are per 1M tokens, configurable; unknown models cost $0 and are
    reported as such, never guessed.
"""

from __future__ import annotations

from arl.core.events import EventType
from arl.core.trace import Trace

# USD per 1M tokens (input, output). These are published list prices; users
# should override for their agreements. "scripted" is the offline reference
# model and is free by construction.
PRICE_TABLE_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "scripted": (0.0, 0.0),
    "mock": (0.0, 0.0),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "o4-mini": (1.10, 4.40),
}


def estimate_tokens(text: str) -> int:
    """The chars/4 heuristic. Only used when a backend reports no usage."""
    return max(1, len(text) // 4)


def price_for(model: str) -> tuple[float, float] | None:
    """Exact match first, then case/prefix-insensitive lookup."""
    key = model.lower()
    if key in PRICE_TABLE_USD_PER_MTOK:
        return PRICE_TABLE_USD_PER_MTOK[key]
    for name, price in PRICE_TABLE_USD_PER_MTOK.items():
        if key.startswith(name) or name.startswith(key):
            return price
    return None


def cost_for_trace(trace: Trace, model: str | None = None) -> tuple[float, str]:
    """(cost_usd, source) for a trace. Source is 'reported' when the backend
    reported usage, else 'estimated'. Models absent from the price table cost
    $0.0 with source 'unpriced' — never an invented number."""
    inp = out = 0
    source = "estimated"
    for event in trace.by_type(EventType.MODEL_RESPONSE):
        usage = event.payload.get("usage") or {}
        inp += int(usage.get("input_tokens", 0) or 0)
        out += int(usage.get("output_tokens", 0) or 0)
        if event.payload.get("token_source") == "reported":
            source = "reported"
    used_model = model
    if used_model is None:
        responses = trace.by_type(EventType.MODEL_RESPONSE)
        used_model = responses[0].name if responses else "scripted"
    price = price_for(used_model or "scripted")
    if price is None:
        return 0.0, "unpriced"
    cost = (inp * price[0] + out * price[1]) / 1_000_000.0
    return cost, source
