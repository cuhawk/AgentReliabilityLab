"""Scenario DSL: schema, loading, suites."""

from arl.scenarios.loader import ScenarioError, load_scenario, load_scenarios
from arl.scenarios.schema import (
    FaultStep,
    FaultType,
    Scenario,
    ScenarioCategory,
    ToolSpec,
)
from arl.scenarios.suite import Suite, load_suite

__all__ = [
    "FaultStep",
    "FaultType",
    "Scenario",
    "ScenarioCategory",
    "ScenarioError",
    "Suite",
    "ToolSpec",
    "load_scenario",
    "load_scenarios",
    "load_suite",
]
