"""Authorization policy engine: risk class → action, plus argument scopes."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from arl.core.events import RiskClass
from arl.scenarios.schema import DEFAULT_POLICY, PolicyAction, ScopeSpec, ToolSpec


def approval_key(tool: str, args: dict[str, Any]) -> str:
    """Stable key binding an approval to (tool, argument values)."""
    blob = json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class PolicyDecision:
    __slots__ = ("action", "allowed", "reason")

    def __init__(self, allowed: bool, action: PolicyAction, reason: str) -> None:
        self.allowed = allowed
        self.action = action
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PolicyDecision(allowed={self.allowed}, action={self.action.value}, reason={self.reason!r})"


class PolicyEngine:
    def __init__(self, rules: dict[RiskClass, PolicyAction]) -> None:
        self._rules = rules

    def action_for(self, risk_class: RiskClass) -> PolicyAction:
        return self._rules.get(risk_class, DEFAULT_POLICY.get(risk_class, PolicyAction.deny))

    def check(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        approvals: dict[str, str],
    ) -> PolicyDecision:
        """Decide whether this call may execute.

        ``approvals`` maps ``approval_key(tool, args)`` → ``"granted"``.
        Unknown tools are denied by the caller before reaching here.
        """
        action = self.action_for(spec.risk_class)
        if action is PolicyAction.allow:
            return PolicyDecision(True, action, "risk class allowed")
        if action is PolicyAction.deny:
            return PolicyDecision(False, action, f"risk class {spec.risk_class.value} is denied by policy")

        # allow_in_scope / require_approval: scope must hold.
        if spec.scope is not None:
            scope_problem = self._scope_violation(spec.scope, args)
            if scope_problem:
                return PolicyDecision(False, action, scope_problem)

        if action is PolicyAction.require_approval:
            key = approval_key(spec.name, args)
            if approvals.get(key) == "granted":
                return PolicyDecision(True, action, "human approval granted for this call")
            return PolicyDecision(False, action, "human approval required before this call")

        return PolicyDecision(True, action, "risk class allowed within scope")

    @staticmethod
    def _scope_violation(scope: ScopeSpec, args: dict[str, Any]) -> str | None:
        value = args.get(scope.field)
        if value is None:
            return f"missing scope field '{scope.field}'"
        if str(value) not in {str(v) for v in scope.allowed}:
            return f"argument '{scope.field}'={value} outside allowed scope {scope.allowed}"
        return None
