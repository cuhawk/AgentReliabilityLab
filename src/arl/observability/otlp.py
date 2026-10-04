"""OTLP exporter factory, isolated so the ``otel`` extra stays optional."""

from __future__ import annotations

from opentelemetry.sdk.trace.export import SpanExporter


def make_otlp_exporter() -> SpanExporter:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "otlp export requires the 'otel' extra: uv pip install 'agent-reliability-lab[otel]'"
        ) from exc
    return OTLPSpanExporter()
