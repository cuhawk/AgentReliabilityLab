"""Adapter tests: the HTTP turn-protocol agent, the OpenAI-compatible
tool-calling loop, and the MCP gateway. All network traffic is served by
httpx.MockTransport — no real endpoints, no credentials, no GPU."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from tests.conftest import REPO_ROOT

from arl.adapters.base import TaskInput
from arl.adapters.http_agent import HttpAgent
from arl.adapters.mcp_gateway import McpGateway
from arl.adapters.openai_agent import OpenAICompatAgent
from arl.environment.env import ToolEnvironment
from arl.faults.injector import FaultInjector
from arl.recording.recorder import TraceRecorder
from arl.runner import RunConfig, run_suite
from arl.scenarios.loader import load_scenarios


def _scenario(scenario_id: str):
    found = load_scenarios([REPO_ROOT / "scenarios" / f"{scenario_id}.yaml"])
    return found if isinstance(found, list) else [found]


def _calc_scenario():
    scenarios = _scenario("happy-calc-add")
    assert len(scenarios) == 1
    return scenarios[0]


def _task_input(scenario) -> TaskInput:
    recorder = TraceRecorder("run-x", scenario.id, "x")
    injector = FaultInjector(scenario, recorder)
    env = ToolEnvironment(scenario, recorder, injector)
    return TaskInput(
        task=scenario.task, scenario_id=scenario.id, tools=env.manifest(),
        env={"variables": env.variables}, state=dict(scenario.env.state),
    )


# ------------------------------------------------------------------ HTTP agent

def test_http_agent_executes_tool_calls_and_finishes() -> None:
    scenario = _calc_scenario()
    turns: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        turns.append(body)
        if len(turns) == 1:
            return httpx.Response(200, json={"tool_calls": [
                {"id": "call-1", "name": "calculator", "args": {"a": 1250.5, "b": 340.0}},
            ], "usage": {"input_tokens": 100, "output_tokens": 20}})
        return httpx.Response(200, json={
            "output": "the sum is 1590.5", "status": "success",
            "usage": {"input_tokens": 150, "output_tokens": 30},
        })

    recorder = TraceRecorder("run-http", scenario.id, "http:test")
    injector = FaultInjector(scenario, recorder)
    env = ToolEnvironment(scenario, recorder, injector)
    agent = HttpAgent("http:test", recorder, env, _task_input(scenario),
                      base_url="http://agent.test",
                      config={"transport": httpx.MockTransport(handler)})
    asyncio.run(agent.start_task())
    trace = asyncio.run(recorder.finish("completed"))

    assert len(turns) == 2  # one turn for the tool call, one for the answer
    assert [c.name for c in trace.tool_calls] == ["calculator"]
    assert trace.tool_calls[0].ok
    assert trace.tool_calls[0].result_value["result"] == pytest.approx(1590.5)
    completion = next(e for e in trace.events if e.type.value == "agent_complete")
    assert completion.payload["output"] == "the sum is 1590.5"


async def test_http_agent_through_the_runner() -> None:
    scenario = _calc_scenario()
    state = {"turns": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["turns"] += 1
        if state["turns"] == 1:
            return httpx.Response(200, json={"tool_calls": [
                {"id": "c1", "name": "calculator", "args": {"a": 1250.5, "b": 340.0}}]})
        return httpx.Response(200, json={"output": "sum: 1590.5", "status": "success"})

    cfg = RunConfig(
        agent_spec="http:agent.test",
        agent_config={"transport": httpx.MockTransport(handler)},
        output_dir=Path(".arl/test-http"),
        report_formats=[],
    )
    run = await run_suite("suite-http", [scenario], cfg)
    result = run.results[0]
    assert result.final_status == "success"
    assert result.tool_calls >= 1


# -------------------------------------------------------------- OpenAI adapter

def test_openai_agent_tool_loop_reports_usage() -> None:
    scenario = _calc_scenario()
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, json={"choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc", "type": "function",
                        "function": {"name": "calculator",
                                     "arguments": json.dumps({"a": 1250.5, "b": 340.0})},
                    }],
                },
                "finish_reason": "tool_calls",
            }], "usage": {"prompt_tokens": 210, "completion_tokens": 25}})
        return httpx.Response(200, json={"choices": [{
            "message": {"role": "assistant", "content": "The result is 1590.5."},
            "finish_reason": "stop",
        }], "usage": {"prompt_tokens": 300, "completion_tokens": 40}})

    recorder = TraceRecorder("run-openai", scenario.id, "openai:test-model")
    injector = FaultInjector(scenario, recorder)
    env = ToolEnvironment(scenario, recorder, injector)
    agent = OpenAICompatAgent(
        "openai:test-model", recorder, env, _task_input(scenario), model="test-model",
        config={"transport": httpx.MockTransport(handler), "base_url": "http://llm.test/v1"},
    )
    asyncio.run(agent.start_task())
    trace = asyncio.run(recorder.finish("completed"))

    assert len(requests) == 2
    assert requests[0]["tools"][0]["function"]["name"] == "calculator"
    completion = next(e for e in trace.events if e.type.value == "agent_complete")
    assert completion.payload["output"] == "The result is 1590.5."
    tokens_in, tokens_out = trace.token_usage()
    assert tokens_in > 0 and tokens_out > 0
    # Usage was reported by the backend, not estimated.
    reported = [e for e in trace.events if e.type.value == "model_response"
                and e.payload.get("token_source") == "reported"]
    assert len(reported) == 2


# ---------------------------------------------------------------- MCP gateway

def _gateway_for(scenario_id: str) -> McpGateway:
    scenario = _scenario(scenario_id)[0]
    recorder = TraceRecorder("run-mcp", scenario.id, "mcp-gateway")
    injector = FaultInjector(scenario, recorder)
    env = ToolEnvironment(scenario, recorder, injector)
    return McpGateway(env)


async def test_mcp_initialize_and_tools_list() -> None:
    gw = _gateway_for("happy-calc-add")
    init = await gw.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["protocolVersion"] == "2024-11-05"
    listing = await gw.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in listing["result"]["tools"]]
    assert names == ["calculator"]
    calc = listing["result"]["tools"][0]
    assert calc["inputSchema"]["required"] == ["a", "b"]


async def test_mcp_call_routes_through_policy_and_faults() -> None:
    gw = _gateway_for("happy-calc-add")
    await gw.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    call = await gw.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "calculator", "arguments": {"a": 2, "b": 3}}})
    assert call["result"]["isError"] is False
    payload = json.loads(call["result"]["content"][0]["text"])
    assert payload["result"] == pytest.approx(5.0)


async def test_mcp_unknown_method_and_denied_tool() -> None:
    gw = _gateway_for("authz-privileged-denied")
    await gw.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    missing = await gw.handle({"jsonrpc": "2.0", "id": 2, "method": "bogus/method"})
    assert missing["error"]["code"] == -32601

    # delete_account is PRIVILEGED under the default policy → denied.
    call = await gw.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "delete_account", "arguments": {"customer_id": "C-104"}}})
    assert call["result"]["isError"] is True


async def test_mcp_requires_initialize_first() -> None:
    gw = _gateway_for("happy-calc-add")
    reply = await gw.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert reply["error"]["code"] == -32002
