"""Environment exceptions. All tool errors carry a stable machine-readable
``kind`` used by evaluators, plus a ``retriable`` hint for agents."""

from __future__ import annotations


class ToolError(Exception):
    """A tool execution failed inside the sandbox."""

    def __init__(self, message: str, kind: str, retriable: bool = False) -> None:
        super().__init__(message)
        self.kind = kind
        self.retriable = retriable


class ToolTimeoutError(ToolError):
    def __init__(self, message: str = "tool timed out") -> None:
        super().__init__(message, kind="timeout", retriable=True)


class ToolHTTPStatusError(ToolError):
    def __init__(self, status: int) -> None:
        super().__init__(f"tool returned HTTP {status}", kind=f"http_{status}", retriable=status >= 429)
        self.status = status


class ToolContentError(ToolError):
    def __init__(self, message: str = "tool returned malformed content") -> None:
        super().__init__(message, kind="malformed_json", retriable=True)


class ToolPermissionError(ToolError):
    def __init__(self, message: str = "permission denied") -> None:
        super().__init__(message, kind="permission_denied", retriable=False)


class PolicyDeniedError(ToolError):
    """The authorization policy refused the call outright."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"policy denied: {reason}", kind="policy_denied", retriable=False)
        self.reason = reason


class ApprovalRequiredError(ToolError):
    """Policy requires human approval before this call can execute."""

    def __init__(self, tool: str) -> None:
        super().__init__(f"tool '{tool}' requires human approval", kind="approval_required", retriable=True)
        self.tool = tool


class StateUnavailableError(ToolError):
    def __init__(self, message: str = "agent state unavailable") -> None:
        super().__init__(message, kind="state_unavailable", retriable=False)


class EnvironmentInterrupted(Exception):
    """The scenario interrupted the run (simulated process kill)."""
