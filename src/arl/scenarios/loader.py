"""Scenario and suite loading with shared tool-library merging."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from arl.scenarios.schema import Scenario

_LIBRARY_FILENAME = "_library.yaml"


class ScenarioError(ValueError):
    """Raised when a scenario/suite file cannot be parsed or validated.

    Message includes the file path; pydantic's error text is appended so the
    YAML author sees exactly which field failed.
    """


def load_tool_library(directory: Path) -> dict[str, dict[str, Any]]:
    """Load ``_library.yaml`` walking upward from *directory* (nearest wins).

    Returns {} when no library file exists, so scenarios are loadable anywhere.
    """
    for candidate in [directory, *directory.parents]:
        lib_file = candidate / _LIBRARY_FILENAME
        if lib_file.is_file():
            try:
                raw = yaml.safe_load(lib_file.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ScenarioError(f"{lib_file}: invalid YAML: {exc}") from exc
            tools = raw.get("tools", {})
            if not isinstance(tools, dict):
                raise ScenarioError(f"{lib_file}: 'tools' must be a mapping")
            return tools
    return {}


def _merge_library_tool(raw: dict[str, Any], library: dict[str, dict[str, Any]], path: Path) -> dict[str, Any]:
    """Merge a ``library: <name>`` reference under inline scenario overrides."""
    lib_name = raw.get("library")
    if not lib_name:
        return raw
    if lib_name not in library:
        known = ", ".join(sorted(library)) or "<empty>"
        raise ScenarioError(f"{path}: library tool '{lib_name}' not found (known: {known})")
    base = dict(library[lib_name])
    overrides = {k: v for k, v in raw.items() if k != "library"}
    merged = {**base, **overrides}
    # Fault lists concatenate rather than replace: library faults are baseline.
    if "faults" in base or "faults" in overrides:
        merged["faults"] = [*base.get("faults", []), *overrides.get("faults", [])]
    return merged


def scenario_from_dict(data: dict[str, Any], path: Path | None = None) -> Scenario:
    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: top level must be a mapping")
    if "scenario" in data and isinstance(data["scenario"], dict):
        data = data["scenario"]
    if path is not None:
        data = {**data, "source_file": str(path)}
        directory = path.parent if path.is_absolute() else path.resolve().parent
        library = load_tool_library(directory)
        tools_raw = data.get("tools", {})
        if isinstance(tools_raw, dict):
            data["tools"] = {
                name: _merge_library_tool(raw, library, path) for name, raw in tools_raw.items()
            }
    try:
        return Scenario.model_validate(data)
    except Exception as exc:
        raise ScenarioError(f"{path or '<memory>'}: {exc}") from exc


def load_scenario(path: str | Path) -> Scenario:
    file = Path(path)
    if not file.is_file():
        raise ScenarioError(f"scenario file not found: {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{file}: invalid YAML: {exc}") from exc
    if raw is None:
        raise ScenarioError(f"{file}: empty scenario file")
    return scenario_from_dict(raw, file)


def load_scenarios(paths: Sequence[str | Path]) -> list[Scenario]:
    """Load a list of scenario files or glob patterns, sorted, deduplicated."""
    files: list[Path] = []
    for entry in paths:
        p = Path(entry)
        if p.is_dir():
            files.extend(sorted(p.glob("*.yaml")))
        else:
            pattern = str(entry)
            matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[") else [p]
            files.extend(matches)
    seen: set[Path] = set()
    out: list[Scenario] = []
    for file in files:
        if file in seen or file.name.startswith("_"):
            continue
        seen.add(file)
        out.append(load_scenario(file))
    return out
