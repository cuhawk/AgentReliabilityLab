"""Shared fixtures for the ARL platform test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from arl.observability.tracing import setup_tracing

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def span_exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    setup_tracing(exporter_instance=exporter)
    yield exporter
    setup_tracing("none")


@pytest.fixture()
def scenario_dir() -> Path:
    return REPO_ROOT / "scenarios"
