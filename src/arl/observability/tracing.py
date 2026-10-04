"""OpenTelemetry integration.

Agent execution is represented as nested spans::

    eval.run
    ├── eval.scenario <id>
    │   ├── agent.task
    │   │   ├── llm.call
    │   │   ├── tool.customer_api
    │   │   ├── retry
    │   │   ├── policy.check
    │   │   └── approval
    │   └── evaluators

Exporters: ``none`` (spans discarded — zero overhead default), ``console``,
``otlp`` (requires the ``otel`` extra). Tests inject an in-memory exporter via
:func:`build_provider`."""

from __future__ import annotations

import atexit
import contextlib
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, set_span_in_context

from arl.core.events import Event, EventType

_SERVICE = "agentreliabilitylab"
_PROVIDER: TracerProvider | None = None


def build_provider(exporter: str = "none", exporter_instance: Any = None) -> TracerProvider:
    """Build an SDK provider. ``exporter_instance`` overrides the exporter
    (used by tests to inject InMemorySpanExporter)."""
    provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE}))
    instance = exporter_instance
    if instance is None and exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        instance = ConsoleSpanExporter()
    if instance is None and exporter == "otlp":
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from arl.observability.otlp import make_otlp_exporter

        provider.add_span_processor(BatchSpanProcessor(make_otlp_exporter()))
        return provider
    if instance is not None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(instance))
    return provider


def setup_tracing(exporter: str = "none", exporter_instance: Any = None) -> TracerProvider:
    """Install the provider globally (idempotent-ish: replaces previous)."""
    global _PROVIDER
    _PROVIDER = build_provider(exporter, exporter_instance)
    with contextlib.suppress(Exception):
        trace.set_tracer_provider(_PROVIDER)
    atexit.register(shutdown_tracing)
    return _PROVIDER


def shutdown_tracing() -> None:
    if _PROVIDER is not None:
        _PROVIDER.shutdown()


def tracer() -> trace.Tracer:
    # Prefer ARL's own provider so repeated setup_tracing() calls (tests,
    # embedded runners) actually take effect; fall back to the ambient global.
    if _PROVIDER is not None:
        return _PROVIDER.get_tracer(_SERVICE)
    return trace.get_tracer(_SERVICE)


class EventSpanTracker:
    """Converts the event stream into live spans. Tool/model/approval spans
    pair a start and end event by id; single events become instant spans. All
    spans are parented explicitly under the ``agent.task`` span so the nesting
    does not depend on ambient context activation across async boundaries."""

    def __init__(self) -> None:
        self._open_tool_spans: dict[str, Span] = {}
        self._open_model_spans: list[Span] = []
        self._open_approval_span: Span | None = None
        self._agent_span: Span | None = None

    def _start_child(self, name: str, attributes: dict[str, Any]) -> Span:
        parent = self._agent_span
        context = set_span_in_context(parent) if parent is not None else None
        return tracer().start_span(name, context=context, attributes=attributes)

    async def __call__(self, event: Event) -> None:
        t = event.type
        span: Span | None
        if t is EventType.AGENT_START:
            self._agent_span = tracer().start_span(
                "agent.task",
                attributes={
                    "arl.task": str(event.payload.get("task", ""))[:200],
                    "arl.scenario": str(event.scenario_id),
                    "arl.agent": str(event.agent_id),
                },
            )
        elif t is EventType.TOOL_REQUEST:
            span = self._start_child(
                f"tool.{event.name}",
                {
                    "arl.tool": str(event.name),
                    "arl.attempt": event.attempt,
                    "arl.args": str(event.payload.get("args", {}))[:400],
                },
            )
            if event.tool_call_id:
                self._open_tool_spans[event.tool_call_id] = span
        elif t is EventType.TOOL_RESULT:
            key = event.tool_call_id or ""
            span = self._open_tool_spans.pop(key) if key in self._open_tool_spans else None
            if span is not None:
                ok = event.payload.get("status") == "ok"
                span.set_attribute("arl.status", str(event.payload.get("status")))
                if event.payload.get("fault"):
                    span.set_attribute("arl.fault", str(event.payload.get("fault")))
                if event.payload.get("error_kind"):
                    span.set_attribute("arl.error_kind", str(event.payload.get("error_kind")))
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR if not ok else trace.StatusCode.OK,
                                 description=str(event.payload.get("error", ""))[:120])
                )
                span.end()
        elif t is EventType.MODEL_REQUEST:
            self._open_model_spans.append(
                self._start_child("llm.call", {"arl.model": str(event.name)})
            )
        elif t is EventType.MODEL_RESPONSE:
            span = self._open_model_spans.pop() if self._open_model_spans else None
            if span is not None:
                usage = event.payload.get("usage") or {}
                span.set_attribute("llm.usage.input_tokens", int(usage.get("input_tokens", 0) or 0))
                span.set_attribute("llm.usage.output_tokens", int(usage.get("output_tokens", 0) or 0))
                span.end()
        elif t is EventType.RETRY:
            span = self._start_child("retry", {"arl.target": str(event.name),
                                               "arl.reason": str(event.payload.get("reason"))})
            span.end()
        elif t is EventType.POLICY_CHECK:
            span = self._start_child("policy.check", {
                "arl.decision": str(event.payload.get("decision")),
                "arl.reason": str(event.payload.get("reason")),
                "arl.tool": str(event.name),
            })
            span.end()
        elif t is EventType.FAULT_INJECTED:
            span = self._start_child(f"fault.{event.payload.get('fault')}", {
                "arl.fault": str(event.payload.get("fault")),
                "arl.target": str(event.name),
                "arl.call_ordinal": int(event.payload.get("call_ordinal", 0) or 0),
            })
            span.end()
        elif t is EventType.HUMAN_APPROVAL_REQUEST:
            self._open_approval_span = self._start_child("approval", {"arl.tool": str(event.name)})
        elif t is EventType.HUMAN_APPROVAL_RESULT:
            span = self._open_approval_span
            self._open_approval_span = None
            if span is not None:
                span.set_attribute("arl.decision", str(event.payload.get("decision")))
                span.end()
        elif t in (EventType.AGENT_COMPLETE, EventType.AGENT_ERROR):
            span = self._agent_span
            self._agent_span = None
            if span is not None:
                if t is EventType.AGENT_ERROR:
                    span.set_status(trace.Status(trace.StatusCode.ERROR,
                                                 description=str(event.payload.get("error", ""))[:120]))
                span.set_attribute("arl.outcome", t.value)
                span.end()

    def close_abandoned(self) -> None:
        for span in self._open_tool_spans.values():
            span.end()
        self._open_tool_spans.clear()
        for span in self._open_model_spans:
            span.end()
        self._open_model_spans.clear()
        if self._open_approval_span is not None:
            self._open_approval_span.end()
            self._open_approval_span = None
        if self._agent_span is not None:
            self._agent_span.end()
            self._agent_span = None
