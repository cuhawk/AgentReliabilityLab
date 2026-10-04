"""SQLite storage for runs, scenario results, events, tool calls, metrics,
violations and baseline comparisons. The eval engine writes here; the
dashboard only reads. (Postgres is an extension point — see Storage ABC.)"""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  suite TEXT,
  agent_id TEXT,
  model TEXT,
  config_json TEXT,
  environment_json TEXT,
  status TEXT,
  scenario_count INTEGER,
  metrics_json TEXT,
  report_paths_json TEXT
);
CREATE TABLE IF NOT EXISTS scenario_results (
  run_id TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  title TEXT,
  category TEXT,
  status TEXT,
  trace_status TEXT,
  final_status TEXT,
  duration_ms REAL,
  metrics_json TEXT,
  checks_json TEXT,
  violations_json TEXT,
  faults_json TEXT,
  trace_text TEXT,
  agent_id TEXT,
  PRIMARY KEY (run_id, scenario_id)
);
CREATE TABLE IF NOT EXISTS events (
  run_id TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  t_offset_ms REAL,
  type TEXT,
  name TEXT,
  tool_call_id TEXT,
  risk_class TEXT,
  attempt INTEGER,
  payload_json TEXT,
  PRIMARY KEY (run_id, scenario_id, seq)
);
CREATE TABLE IF NOT EXISTS tool_calls (
  run_id TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  tool_call_id TEXT NOT NULL,
  name TEXT,
  attempt INTEGER,
  args_json TEXT,
  status TEXT,
  error_kind TEXT,
  fault TEXT,
  latency_ms REAL,
  result_json TEXT,
  PRIMARY KEY (run_id, scenario_id, tool_call_id)
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL,
  scenario_id TEXT,
  name TEXT NOT NULL,
  value REAL,
  unit TEXT,
  ci_low REAL,
  ci_high REAL,
  n INTEGER,
  source TEXT,
  PRIMARY KEY (run_id, scenario_id, name)
);
CREATE TABLE IF NOT EXISTS violations (
  run_id TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  seq INTEGER,
  kind TEXT,
  tool TEXT,
  detail TEXT
);
CREATE TABLE IF NOT EXISTS comparisons (
  id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  baseline_run_id TEXT,
  candidate_run_id TEXT,
  verdict TEXT,
  table_json TEXT,
  thresholds_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_results_run ON scenario_results(run_id);
"""


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class Storage(ABC):
    """Persistence boundary. SQLite ships; Postgres would implement this."""

    @abstractmethod
    def save_run(self, run: dict[str, Any]) -> None: ...
    @abstractmethod
    def save_scenario_result(self, run_id: str, result: Any) -> None: ...
    @abstractmethod
    def save_events(self, run_id: str, scenario_id: str, events: list[Any]) -> None: ...
    @abstractmethod
    def save_tool_calls(self, run_id: str, scenario_id: str, calls: list[Any]) -> None: ...
    @abstractmethod
    def save_metrics(self, run_id: str, metrics: dict[str, Any], scenario_id: str | None = None) -> None: ...
    @abstractmethod
    def save_violations(self, run_id: str, scenario_id: str, violations: list[Any]) -> None: ...
    @abstractmethod
    def save_comparison(self, comparison: dict[str, Any]) -> None: ...
    @abstractmethod
    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]: ...
    @abstractmethod
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    @abstractmethod
    def get_scenario_results(self, run_id: str) -> list[dict[str, Any]]: ...
    @abstractmethod
    def get_events(self, run_id: str, scenario_id: str) -> list[dict[str, Any]]: ...
    @abstractmethod
    def get_scenario_metrics(self, run_id: str) -> list[dict[str, Any]]: ...
    @abstractmethod
    def list_comparisons(self, limit: int = 20) -> list[dict[str, Any]]: ...


class SqliteStorage(Storage):
    def __init__(self, path: str | Path) -> None:
        import threading

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the dashboard serves requests from worker
        # threads; the lock serializes access instead.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- writes -----------------------------------------------------------

    def save_run(self, run: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run["run_id"], run.get("created_at", time.time()), run.get("suite"),
                    run.get("agent_id"), run.get("model"), _dump(run.get("config", {})),
                    _dump(run.get("environment", {})), run.get("status", "completed"),
                    run.get("scenario_count", 0), _dump(run.get("metrics", {})),
                    _dump(run.get("report_paths", {})),
                ),
            )
            self._conn.commit()

    def save_scenario_result(self, run_id: str, result: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO scenario_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, result.scenario_id, result.title, result.category, result.status,
                    result.trace_status, result.final_status, result.duration_ms,
                    _dump(result.metrics), _dump([c.model_dump() for c in result.checks]),
                    _dump([v.model_dump() for v in result.violations]), _dump(result.faults),
                    result.trace_text, result.agent_id,
                ),
            )
            self._conn.commit()

    def save_events(self, run_id: str, scenario_id: str, events: list[Any]) -> None:
        rows = [
            (run_id, scenario_id, e.seq, e.t_offset_ms, e.type.value, e.name, e.tool_call_id,
             e.risk_class.value if e.risk_class else None, e.attempt, _dump(e.payload))
            for e in events
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )
            self._conn.commit()

    def save_tool_calls(self, run_id: str, scenario_id: str, calls: list[Any]) -> None:
        rows = [
            (run_id, scenario_id, c.tool_call_id, c.name, c.attempt, _dump(c.args), c.status,
             c.error_kind, c.fault, c.latency_ms,
             _dump(c.result_value) if c.result_value is not None else None)
            for c in calls
        ]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._conn.commit()

    def save_metrics(self, run_id: str, metrics: dict[str, Any], scenario_id: str | None = None) -> None:
        rows = [
            (run_id, scenario_id, name, m.value, m.unit, m.ci_low, m.ci_high, m.n, m.source)
            for name, m in metrics.items()
        ]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self._conn.commit()

    def save_violations(self, run_id: str, scenario_id: str, violations: list[Any]) -> None:
        rows = [(run_id, scenario_id, v.seq, v.kind, v.tool, v.detail) for v in violations]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO violations VALUES (?,?,?,?,?,?)", rows)
            self._conn.commit()

    def save_comparison(self, comparison: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO comparisons VALUES (?,?,?,?,?,?,?)",
                (
                    comparison["comparison_id"], comparison.get("created_at", time.time()),
                    comparison.get("baseline_run_id"), comparison.get("candidate_run_id"),
                    comparison.get("verdict"), _dump(comparison.get("table", [])),
                    _dump(comparison.get("thresholds", [])),
                ),
            )
            self._conn.commit()

    # --- reads --------------------------------------------------------------

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(query, params).fetchall()]

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM runs WHERE id = ?", (run_id,))
        return rows[0] if rows else None

    def get_scenario_results(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM scenario_results WHERE run_id = ? ORDER BY scenario_id", (run_id,))

    def get_events(self, run_id: str, scenario_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM events WHERE run_id = ? AND scenario_id = ? ORDER BY seq",
            (run_id, scenario_id),
        )

    def get_tool_calls(self, run_id: str, scenario_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM tool_calls WHERE run_id = ? AND scenario_id = ? ORDER BY tool_call_id",
            (run_id, scenario_id),
        )

    def get_scenario_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM metrics WHERE run_id = ?", (run_id,))

    def get_run_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM metrics WHERE run_id = ? AND scenario_id IS NULL", (run_id,))

    def list_comparisons(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM comparisons ORDER BY created_at DESC LIMIT ?", (limit,))
