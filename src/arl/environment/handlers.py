"""Mock tool handlers. Everything runs inside the sandbox — effect tools write
to an in-process outbox, nothing touches a real network. Handlers are pure
functions of (spec, args, HandlerContext) so they are trivially testable."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from email.utils import parseaddr
from typing import Any
from urllib.parse import urlparse

from arl.scenarios.schema import ToolSpec

Handler = Callable[[ToolSpec, dict[str, Any], "HandlerContext"], Awaitable[Any]]


class HandlerContext:
    """Narrow view of the environment a handler may touch."""

    def __init__(
        self,
        *,
        allowlisted_domains: list[str],
        http_ordinal: int,
        record_effect: Callable[..., None],
    ) -> None:
        self.allowlisted_domains = allowlisted_domains
        self.http_ordinal = http_ordinal
        self._record_effect = record_effect

    def record_effect(self, **kwargs: Any) -> None:
        self._record_effect(**kwargs)


def _host_of(target: str | None) -> str | None:
    if not target:
        return None
    if "://" in target or target.startswith("//"):
        host = urlparse(target if "://" in target else "https:" + target).hostname
        return host.lower() if host else None
    if "@" in target:
        # Email address (or bare user@host) — take the domain part.
        _, addr = parseaddr(target)
        addr = addr or target
        if "@" in addr:
            return addr.rsplit("@", 1)[-1].lower()
    return target.strip().lower()


def _is_allowlisted(host: str | None, allowlist: list[str]) -> bool:
    if host is None:
        return False
    return any(host == d or host.endswith("." + d) for d in allowlist)


# --- data --------------------------------------------------------------------


async def run_data(spec: ToolSpec, args: dict[str, Any], ctx: HandlerContext) -> Any:
    key_field = str(spec.params.get("key", "id"))
    records: list[dict[str, Any]] = spec.params.get("records", [])
    wanted = args.get(key_field)
    for record in records:
        if str(record.get(key_field)) == str(wanted):
            return {"found": True, "record": dict(record), key_field: wanted}
    return {"found": False, "record": None, key_field: wanted}


# --- compute -----------------------------------------------------------------


async def run_compute(spec: ToolSpec, args: dict[str, Any], ctx: HandlerContext) -> Any:
    function = str(spec.params.get("function"))
    if function == "add":
        return {"function": "add", "result": float(args["a"]) + float(args["b"])}
    if function == "multiply":
        return {"function": "multiply", "result": float(args["a"]) * float(args["b"])}
    if function == "percentage":
        return {"function": "percentage", "result": round(float(args["value"]) * float(args["pct"]) / 100.0, 4)}
    if function == "currency_convert":
        rates: dict[str, float] = spec.params.get("rates", {})
        base = str(args.get("from", "USD")).upper()
        quote = str(args.get("to", "USD")).upper()
        if base not in rates or quote not in rates:
            return {"function": "currency_convert", "error": f"unknown currency pair {base}/{quote}"}
        converted = round(float(args["amount"]) * rates[quote] / rates[base], 4)
        return {"function": "currency_convert", "result": converted, "unit": quote}
    return {"error": f"unknown compute function '{function}'"}


# --- retrieve ----------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


async def run_retrieve(spec: ToolSpec, args: dict[str, Any], ctx: HandlerContext) -> Any:
    corpus: list[dict[str, Any]] = spec.params.get("corpus", [])
    top_k = int(spec.params.get("top_k", 3))
    query = str(args.get("query", ""))
    q_tokens = set(_WORD_RE.findall(query.lower()))
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in corpus:
        text_tokens = set(_WORD_RE.findall((str(doc.get("title", "")) + " " + str(doc.get("content", ""))).lower()))
        overlap = len(q_tokens & text_tokens)
        score = overlap * (3.0 if doc.get("relevance", "low") == "high" else 1.0)
        if overlap:
            scored.append((score, doc))
    scored.sort(key=lambda pair: -pair[0])
    docs = [dict(doc) for _, doc in scored[:top_k]]
    return {"query": query, "documents": docs}


# --- effect ------------------------------------------------------------------

_EFFECT_TARGET_FIELD = {
    "email": "to",
    "webhook": "url",
    "transfer": "account",
    "delete_record": "record_id",
}


async def run_effect(spec: ToolSpec, args: dict[str, Any], ctx: HandlerContext) -> Any:
    effect_type = str(spec.params["effect_type"])
    target_field = _EFFECT_TARGET_FIELD.get(effect_type, "target")
    target = args.get(target_field)
    host = _host_of(str(target) if target is not None else None)
    allowlisted = _is_allowlisted(host, ctx.allowlisted_domains)
    ctx.record_effect(
        tool=spec.name,
        effect_type=effect_type,
        target=str(target) if target is not None else None,
        target_host=host,
        target_allowlisted=allowlisted,
        args=dict(args),
    )
    return {
        "effect": effect_type,
        "target": target,
        "recorded": True,
        "target_allowlisted": allowlisted,
    }


# --- http --------------------------------------------------------------------


async def run_http(spec: ToolSpec, args: dict[str, Any], ctx: HandlerContext) -> Any:
    script: list[dict[str, Any]] = spec.params.get("script", [])
    if not script:
        return {"status": 500, "body": {"error": "no script"}}
    entry = script[min(ctx.http_ordinal, len(script) - 1)]
    host = spec.params.get("host", "api.company.example")
    path = str(args.get("path", "/")).strip() or "/"
    return {"status": int(entry.get("status", 200)), "body": entry.get("body"), "url": f"https://{host}{path}"}


HANDLERS: dict[str, Handler] = {
    "data": run_data,
    "compute": run_compute,
    "retrieve": run_retrieve,
    "effect": run_effect,
    "http": run_http,
}
