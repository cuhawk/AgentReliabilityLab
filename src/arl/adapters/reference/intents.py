"""Deterministic intent parsing: turns a natural-language task into a plan of
typed steps. This is the "brain" of the reference agent's scripted model
client — a small, honest rule engine, not an LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from arl.environment.env import ToolManifest

_ENTITY_RE = re.compile(r"\b([A-Z]{1,6}-\d+)\b")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class PlanStep:
    kind: str  # lookup | retrieve | compute | notify | destroy | transfer | write | compose
    entity: str | None = None
    query: str = ""
    tool_hint: str | None = None
    effect: str | None = None
    target: str | None = None
    numbers: list[float] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)  # fields the answer should mention
    execute: bool = True  # False for prepare-only destructive steps


def entities_of(task: str) -> list[str]:
    return _ENTITY_RE.findall(task)


def _topic_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", text.lower())}


def score_tool(tool: ToolManifest, tokens: set[str]) -> float:
    """Keyword overlap between the task and the tool's name/description/args."""
    tool_tokens = _topic_tokens(tool.name.replace("_", " ") + " " + tool.description)
    for arg in tool.args_schema.values():
        tool_tokens |= _topic_tokens(str(arg.get("description", "")))
    return len(tokens & tool_tokens)


def build_plan(task: str, tools: list[ToolManifest]) -> list[PlanStep]:
    """Derive an ordered plan from the task text. Deterministic keyword rules."""
    steps: list[PlanStep] = []
    lower = task.lower()
    entities = entities_of(task)
    tokens = _topic_tokens(task)

    def best_data_tool(hint_tokens: set[str]) -> ToolManifest | None:
        data_tools = [t for t in tools if t.risk_class.value in ("read", "write") and t.name not in ("docs",)]
        scored = sorted(((score_tool(t, hint_tokens | tokens), t) for t in data_tools), key=lambda p: -p[0])
        return scored[0][1] if scored and scored[0][0] > 0 else (data_tools[0] if data_tools else None)

    wants_retrieval = any(w in lower for w in ("search", "documentation", "policy", "knowledge", "guide", "manual"))
    wants_lookup = any(w in lower for w in (
        "retrieve", "look up", "lookup", "fetch", "get", "status", "details", "find",
        # "summarize the account" style tasks imply fetching the record first.
        "summarize", "profile", "history", "report on",
    ))
    wants_compute = any(w in lower for w in
                        ("calculate", "convert", "total", "sum", "multiply", "percentage", "how much"))
    wants_email = any(w in lower for w in ("send", "email", "notify", "mail"))
    wants_webhook = any(w in lower for w in ("post", "webhook", "publish"))
    wants_transfer = any(w in lower for w in ("transfer", " wire ", "pay", "charge"))
    wants_destroy = any(w in lower for w in (
        "delete", "deletion", "deleting", "remove", "cancel", "deactivate", "purge", "decommission",
    ))
    wants_write = any(w in lower for w in ("write", "save a note", "update the note", "record a note"))
    prepare_only = "not execute" in lower or "without executing" in lower or "do not run" in lower

    if wants_retrieval:
        steps.append(PlanStep(kind="retrieve", query=task))
    if wants_lookup and entities:
        for entity in entities:
            idx = task.find(entity)
            hint = task[max(0, idx - 40): idx] if idx > 0 else ""
            steps.append(PlanStep(kind="lookup", entity=entity, tool_hint=hint))
    elif wants_lookup:
        steps.append(PlanStep(kind="lookup", query=task))

    if wants_compute:
        numbers = [float(n) for n in _NUMBER_RE.findall(task)]
        steps.append(PlanStep(kind="compute", query=task, numbers=numbers))
    if wants_email:
        match = re.search(r"to\s+([\w.+-]+@[\w.-]+)", task)
        steps.append(PlanStep(kind="notify", effect="email", target=match.group(1) if match else None))
    if wants_webhook:
        match = re.search(r"(https?://\S+)", task)
        steps.append(PlanStep(kind="notify", effect="webhook", target=match.group(1) if match else None))
    if wants_transfer:
        match = re.search(r"(?:to|account)\s+([A-Za-z0-9-]+)", task)
        steps.append(PlanStep(kind="transfer", effect="transfer", target=match.group(1) if match else None,
                              numbers=[float(n) for n in _NUMBER_RE.findall(task)]))
    if wants_destroy and entities:
        for entity in entities:
            steps.append(PlanStep(kind="destroy", effect="delete_record", entity=entity, execute=not prepare_only))
    elif wants_destroy:
        steps.append(PlanStep(kind="destroy", effect="delete_record", execute=not prepare_only))

    if wants_write and entities:
        steps.append(PlanStep(kind="write", entity=entities[0]))

    steps.append(PlanStep(kind="compose"))
    return steps


def pick_tool_for_step(step: PlanStep, tools: list[ToolManifest], *, strategy: str) -> ToolManifest | None:
    """Choose the tool executing a plan step. ``strategy``:
    ``best_score`` (guarded) vs ``first_match`` (weak routing bug)."""
    if step.kind == "retrieve":
        candidates = [t for t in tools if t.name in ("docs", "knowledge_base", "search", "retriever")]
        candidates = candidates or [t for t in tools if t.risk_class is not None and "doc" in t.description.lower()]
        return candidates[0] if candidates else None
    if step.kind in ("notify", "destroy", "transfer", "write"):
        effect = step.effect or ""
        by_effect = [t for t in tools if effect and effect in str(t.args_schema.keys())]
        by_kind = [t for t in tools if t.risk_class.value in ("external_side_effect", "privileged")]
        pool = by_effect or by_kind
        if step.kind == "destroy":
            destroy = [t for t in tools if "delete" in t.name or "delete" in t.description.lower()]
            pool = destroy or pool
        if step.kind == "notify":
            email = [t for t in tools if "email" in t.name or "mail" in t.name]
            pool = email or pool
        if step.kind == "write":
            writes = [t for t in tools if t.risk_class.value == "write"]
            pool = writes or pool
        return pool[0] if pool else None
    if step.kind == "compute":
        compute = [t for t in tools if "calc" in t.name or "convert" in t.name or "comput" in t.description.lower()]
        return compute[0] if compute else None
    # lookup: data-ish tools
    data_tools = [
        t for t in tools
        if t.risk_class.value in ("read", "write") and t.name not in ("docs", "knowledge_base")
    ]
    if not data_tools:
        return None
    text = (step.query or "") + " " + (step.entity or "").replace("-", " ") + " " + (step.tool_hint or "")
    tokens = _topic_tokens(text)
    if strategy == "first_match":
        return data_tools[0]
    scored = sorted(((score_tool(t, tokens), t) for t in data_tools), key=lambda p: -p[0])
    return scored[0][1] if scored else data_tools[0]
