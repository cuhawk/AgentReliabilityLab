"""Example: ARL's scenario tools exposed as an MCP server.

ARL acts as the *tool side* of MCP: it serves the scenario's mock tools over
JSON-RPC 2.0 (stdio), and any MCP-capable agent connects as the client. Every
tools/call crosses the ARL gateway — faults, authorization policy, canary
secrets and trace recording all apply.

Run it (interactive; Ctrl+C / EOF to stop)::

    arl mcp-serve scenarios/demo-happy-lookup.yaml

Then drive it from another terminal by speaking MCP JSON-RPC on stdin, e.g.::

    {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
    {"jsonrpc":"2.0","id":2,"method":"tools/list"}
    {"jsonrpc":"2.0","id":3,"method":"tools/call",
     "params":{"name":"customer_api","arguments":{"customer_id":"C-101"}}}

This script shows the same thing in-process, without stdio.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from arl.adapters.mcp_gateway import McpGateway
from arl.environment.env import ToolEnvironment
from arl.faults.injector import FaultInjector
from arl.recording.recorder import TraceRecorder
from arl.scenarios.loader import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    scenario = load_scenario(REPO_ROOT / "scenarios" / "demo-happy-lookup.yaml")
    recorder = TraceRecorder("run-mcp-example", scenario.id, "mcp-gateway")
    env = ToolEnvironment(scenario, recorder, FaultInjector(scenario, recorder))
    gateway = McpGateway(env)

    def rpc(method: str, params: dict, request_id: int = 0) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}

    init = await gateway.handle(rpc("initialize", {}, 1))
    print("initialize ->", init["result"]["serverInfo"])

    listing = await gateway.handle(rpc("tools/list", {}, 2))
    print("tools      ->", [t["name"] for t in listing["result"]["tools"]])

    call = await gateway.handle(rpc(
        "tools/call", {"name": "customer_api", "arguments": {"customer_id": "C-101"}}, 3))
    print("tools/call ->", json.dumps(json.loads(call["result"]["content"][0]["text"]), indent=2))

    trace = await recorder.finish("completed")
    print(f"\nrecorded {len(trace.events)} events across {len(trace.tool_calls)} tool call(s);"
          " the same trace the evaluators would score.")


if __name__ == "__main__":
    asyncio.run(main())
