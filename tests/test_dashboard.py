"""Dashboard tests: read-only endpoints over a populated SQLite store."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from tests.conftest import REPO_ROOT

from arl.dashboard.app import create_app
from arl.recording.storage import SqliteStorage
from arl.runner import RunConfig, run_suite
from arl.scenarios.loader import load_scenarios


async def _populate(tmp_path: Path) -> SqliteStorage:
    scenarios = load_scenarios([str(REPO_ROOT / "scenarios" / "happy-calc-add.yaml")])
    if not isinstance(scenarios, list):
        scenarios = [scenarios]
    storage = SqliteStorage(tmp_path / "lab.db")
    cfg = RunConfig(output_dir=tmp_path / "out", report_formats=[])
    await run_suite("suite-dashboard", scenarios, cfg, storage=storage)
    return storage


async def test_dashboard_endpoints(tmp_path: Path) -> None:
    storage = await _populate(tmp_path)
    try:
        client = TestClient(create_app(tmp_path / "lab.db", storage=storage))

        page = client.get("/")
        assert page.status_code == 200
        assert "AgentReliabilityLab" in page.text

        runs = client.get("/api/runs").json()
        assert len(runs) == 1
        run_id = runs[0]["run_id"]
        assert runs[0]["metrics"]["suite_pass_rate"]["value"] == 1.0

        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["results"][0]["scenario_id"] == "happy-calc-add"
        assert detail["results"][0]["status"] == "pass"

        scenario = client.get(f"/api/runs/{run_id}/scenarios/happy-calc-add").json()
        assert scenario["status"] == "pass"
        assert any(c["name"] == "calculator" for c in scenario["tool_calls"])
        assert "TOOL_REQUEST" in scenario["trace_text"]

        assert client.get("/api/runs/missing").status_code == 404
        assert client.get(
            f"/api/runs/{run_id}/scenarios/not-a-scenario").status_code == 404
        assert client.get("/api/comparisons").json() == []
    finally:
        storage.close()
