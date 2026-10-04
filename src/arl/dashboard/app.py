"""ARL dashboard: a read-only FastAPI app over stored runs (``arl serve``).

The dashboard never writes and never runs evaluations; the eval engine owns
the database. Serves a single self-contained page (static/index.html, no
build step) backed by JSON endpoints over :class:`SqliteStorage` reads."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from arl.recording.storage import Storage


def _loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def create_app(db_path: str | Path, storage: Storage | None = None) -> FastAPI:
    """Build the dashboard app. ``storage`` is injectable for tests."""
    from arl.recording.storage import SqliteStorage

    store = storage if storage is not None else SqliteStorage(db_path)
    app = FastAPI(title="AgentReliabilityLab dashboard", docs_url=None, redoc_url=None)

    def _run_dict(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": row["id"],
            "created_at": row["created_at"],
            "suite": row["suite"],
            "agent_id": row["agent_id"],
            "model": row["model"],
            "status": row["status"],
            "scenario_count": row["scenario_count"],
            "metrics": _loads(row["metrics_json"]),
            "report_paths": _loads(row["report_paths_json"]),
        }

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_file = resources.files("arl.dashboard").joinpath("static/index.html")
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    @app.get("/api/runs")
    async def runs(limit: int = 50) -> list[dict[str, Any]]:
        return [_run_dict(r) for r in store.list_runs(limit=limit)]

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str) -> dict[str, Any]:
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
        results = []
        for r in store.get_scenario_results(run_id):
            results.append({
                "scenario_id": r["scenario_id"],
                "title": r["title"],
                "category": r["category"],
                "status": r["status"],
                "final_status": r["final_status"],
                "trace_status": r["trace_status"],
                "duration_ms": r["duration_ms"],
                "agent_id": r["agent_id"],
                "metrics": _loads(r["metrics_json"]),
                "checks": _loads(r["checks_json"]),
                "faults": _loads(r["faults_json"]),
            })
        return {**_run_dict(row), "results": results}

    @app.get("/api/runs/{run_id}/scenarios/{scenario_id}")
    async def scenario_detail(run_id: str, scenario_id: str) -> dict[str, Any]:
        matches = [r for r in store.get_scenario_results(run_id) if r["scenario_id"] == scenario_id]
        if not matches:
            raise HTTPException(status_code=404, detail="scenario not found in this run")
        row = matches[0]
        return {
            "scenario_id": row["scenario_id"],
            "title": row["title"],
            "status": row["status"],
            "final_status": row["final_status"],
            "duration_ms": row["duration_ms"],
            "metrics": _loads(row["metrics_json"]),
            "checks": _loads(row["checks_json"]),
            "violations": _loads(row["violations_json"]),
            "faults": _loads(row["faults_json"]),
            "trace_text": row["trace_text"],
            "tool_calls": [
                {
                    "name": c["name"],
                    "attempt": c["attempt"],
                    "status": c["status"],
                    "error_kind": c["error_kind"],
                    "fault": c["fault"],
                    "latency_ms": c["latency_ms"],
                    "args": _loads(c["args_json"]),
                    "result": _loads(c["result_json"]),
                }
                for c in store.get_tool_calls(run_id, scenario_id)
            ],
        }

    @app.get("/api/comparisons")
    async def comparisons(limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "comparison_id": c["id"],
                "created_at": c["created_at"],
                "baseline_run_id": c["baseline_run_id"],
                "candidate_run_id": c["candidate_run_id"],
                "verdict": c["verdict"],
                "table": _loads(c["table_json"]),
            }
            for c in store.list_comparisons(limit=limit)
        ]

    return app
