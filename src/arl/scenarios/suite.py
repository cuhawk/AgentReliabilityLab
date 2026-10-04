"""Suite files: named collections of scenarios."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from arl.scenarios.loader import ScenarioError, load_scenarios


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str = ""
    scenario_globs: list[str] = Field(default_factory=list)
    source_file: str | None = Field(default=None, exclude=True)


def load_suite(path: str | Path) -> tuple[Suite, list]:
    """Load a suite file and expand its globs into concrete scenarios.

    Returns (suite, scenarios). Glob paths are resolved relative to the suite
    file. Unknown patterns raise :class:`ScenarioError`.
    """
    import yaml

    file = Path(path)
    if not file.is_file():
        raise ScenarioError(f"suite file not found: {file}")
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if isinstance(raw.get("suite"), dict):
        raw = raw["suite"]
    elif isinstance(raw.get("suite"), str):
        # ``suite: <id>`` shorthand at the top level.
        suite_id = raw.pop("suite")
        raw = {**raw, "id": suite_id}
    if "scenarios" in raw and "scenario_globs" not in raw:
        globs = raw.pop("scenarios")
        raw = {**raw, "scenario_globs": globs}
    if isinstance(raw.get("scenario_globs"), str):
        # Accept a single glob string for convenience.
        raw = {**raw, "scenario_globs": [raw["scenario_globs"]]}
    suite = Suite.model_validate({**raw, "source_file": str(file)})

    base = file.parent
    expanded: list[str] = []
    for pattern in suite.scenario_globs:
        matches = sorted(base.glob(pattern))
        if not matches:
            raise ScenarioError(f"{file}: scenario pattern '{pattern}' matched no files")
        expanded.extend(str(m) for m in matches if not m.name.startswith("_"))

    scenarios = load_scenarios(expanded)
    if not scenarios:
        raise ScenarioError(f"{file}: suite contains no scenarios")
    return suite, scenarios
