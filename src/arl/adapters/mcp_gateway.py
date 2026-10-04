"""ARL as an MCP server: expose scenario tools to any MCP-capable agent.

This is the practical MCP integration (see docs/ADAPTERS.md): an MCP-native
agent connects to this gateway over stdio, discovers the scenario's tools via
``tools/list``, and calls them via ``tools/call``. Every call is routed through
the :class:`ToolEnvironment`, so fault injection, the authorization policy,
the violation log, canary secrets and trace recording all apply exactly as
they do for built-in adapters.

Implemented subset of the Model Context Protocol (JSON-RPC 2.0 over stdio):

    initialize            → protocol version + server capabilities
    notifications/initialized   (acknowledged, no reply)
    tools/list            → {"tools": [...]} in MCP tool schema
    tools/call            → {"content": [{"type": "text", ...}], "isError": bool}
    ping                  → {}

Everything destructive stays inside the sandbox: effect tools append to the
outbox, nothing leaves the process."""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
_SERVER_INFO = {"name": "agentreliabilitylab", "version": "0.1.0"}


class McpGateway:
    """JSON-RPC 2.0 request handler over a ToolEnvironment."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self._initialized = False

    async def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = str(request.get("method", ""))
        request_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            self._initialized = True
            return self._result(request_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": _SERVER_INFO,
            })
        if method.startswith("notifications/"):
            return None  # notifications are never answered
        if not self._initialized:
            return self._error(request_id, -32002, "server not initialized")
        if method == "tools/list":
            return self._result(request_id, {"tools": [manifest_to_mcp_tool(m) for m in self._env.manifest()]})
        if method == "tools/call":
            return await self._tools_call(request_id, params)
        if method == "ping":
            return self._result(request_id, {})
        return self._error(request_id, -32601, f"method not found: {method}")

    async def _tools_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        from arl.environment.errors import ToolError

        name = str(params.get("name", ""))
        args = dict(params.get("arguments") or {})
        try:
            result = await self._env.call_tool(name, args, tool_call_id=f"mcp-{name}")
        except ToolError as exc:
            return self._result(request_id, {
                "content": [{"type": "text", "text": json.dumps(
                    {"error_kind": exc.kind, "error": str(exc)})}],
                "isError": True,
            })
        return self._result(request_id, {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
            "isError": False,
        })

    @staticmethod
    def _result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def manifest_to_mcp_tool(manifest: Any) -> dict[str, Any]:
    """Map an ARL tool manifest onto an MCP tool definition."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for arg_name, arg in manifest.args_schema.items():
        properties[arg_name] = {"type": str(arg.get("type", "string")),
                                "description": str(arg.get("description", ""))}
        if arg.get("required"):
            required.append(arg_name)
    return {
        "name": manifest.name,
        "description": manifest.description,
        "inputSchema": {"type": "object", "properties": properties, "required": required},
    }


def serve_stdio(env: Any) -> None:
    """Read JSON-RPC requests from stdin (one JSON object per line), write
    responses to stdout. Runs until stdin closes. Used by ``arl mcp-serve``.
    Deliberately synchronous: MCP stdio is strict request/response, and a
    blocking read loop works identically on Windows and POSIX."""
    import asyncio

    gateway = McpGateway(env)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response: dict[str, Any] | None = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        else:
            response = asyncio.run(gateway.handle(request))
        if response is not None:
            sys.stdout.write(json.dumps(response, default=str) + "\n")
            sys.stdout.flush()
