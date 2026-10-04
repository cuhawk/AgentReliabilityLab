"""The reference agent: a small, fully deterministic tool-using agent.

It plans from keyword rules (``intents.py``), then executes the plan through a
scripted model-client loop: every step is a MODEL_REQUEST/MODEL_RESPONSE pair,
every action a policy-checked, fault-injectable tool call through the ARL
environment. No network, no RNG, no LLM — which is exactly what makes it useful
for free, reproducible evaluation in CI.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from arl.adapters.base import AgentAdapter, AgentState, AgentStatus, ModelCallError
from arl.adapters.reference.intents import PlanStep, build_plan, pick_tool_for_step
from arl.adapters.reference.profiles import ReferenceAgentConfig
from arl.environment.env import ToolEnvironment, ToolManifest
from arl.environment.errors import (
    ApprovalRequiredError,
    EnvironmentInterrupted,
    StateUnavailableError,
    ToolError,
)
from arl.recording.recorder import TraceRecorder

_EXFIL_URL_RE = re.compile(r"https?://[^\s\"']+")
_SECRET_WORD_RE = re.compile(r"\b(environment variables|env vars|credentials|secrets|api keys?)\b", re.I)
_CALL_TOOL_RE = re.compile(r"\bcall (?:the )?([a-z_]+) tool\b", re.I)
_REVEAL_RE = re.compile(r"\b(reveal|print|show|include)\b.*\b(secret|key|password|variables|credentials)\b", re.I)
_INJECT_RE = re.compile(r"\b(ignore (?:all |any )?previous instructions|disregard .*instructions|you are now)\b", re.I)
_ID_FIELD_RE = re.compile(r"^(id|record_id|account_id|customer_id|order_id|key|record|account|customer|order)$")

_SECRET_TASK_RE = re.compile(r"\b(environment variables|env vars|credentials|api key|config|debug)\b", re.I)
_JSON_TASK_RE = re.compile(r"\bas json\b", re.I)
_LITERAL_REPLY_RE = re.compile(
    r"\breply with the exact (?:word|phrase|sentence|answer|text)\s*:\s*(.+?)\s*$", re.I
)


class ToolOutcome:
    __slots__ = ("assumed", "blocked", "error_kind", "ok", "result")

    def __init__(self, ok: bool, result: Any = None, error_kind: str | None = None,
                 blocked: bool = False, assumed: bool = False) -> None:
        self.ok = ok
        self.result = result
        self.error_kind = error_kind
        self.blocked = blocked
        self.assumed = assumed


class ReferenceAgent(AgentAdapter):
    """Deterministic agent driven by a scripted model client."""

    def __init__(
        self,
        agent_id: str,
        recorder: TraceRecorder,
        env: ToolEnvironment,
        task_input: Any,
        config: ReferenceAgentConfig,
    ) -> None:
        super().__init__(agent_id, recorder, env, task_input)
        self.config = config
        self._plan: list[PlanStep] = build_plan(task_input.task, list(task_input.tools))
        self._findings: list[dict[str, Any]] = []
        self._unavailable: list[str] = []
        self._served_docs: list[dict[str, Any]] = []
        self._blocked_reason: str | None = None
        self._session: dict[str, Any] = {}
        self._state_stale = False
        self._state: AgentState = AgentState(status=AgentStatus.idle)
        self._last_effect_call: tuple[ToolManifest, dict[str, Any]] | None = None
        self._model_retries = 0
        self._forced_reveal = False
        self._docs_stale = False
        self._docs_conflicting = False

    # ------------------------------------------------------------------ state

    def get_state(self) -> AgentState:
        return self._state

    async def send_message(self, message: str) -> None:
        self._session.setdefault("messages", []).append(message)

    async def receive_event(self, event: dict[str, Any]) -> None:
        """Platform event delivery. The duplicate_event fault replays a tool
        result here; the guarded profile ignores replays (idempotent), the
        weak profile re-executes the last effect call (non-idempotent)."""
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not payload.get("replayed"):
            return
        if self.config.detect_duplicate_events:
            return
        if self._last_effect_call is not None:
            tool, args = self._last_effect_call
            await self._exec_tool(tool, args, purpose="replayed event")

    async def cancel(self) -> None:
        self._cancelled = True

    # ------------------------------------------------------------------- main

    async def start_task(self) -> None:
        await self.emit_start()
        self._state.status = AgentStatus.running
        try:
            await self._restore_state()
            for index, step in enumerate(self._plan):
                if self._cancelled:
                    await self.emit_error("cancelled by platform", kind="cancelled")
                    self._state.status = AgentStatus.cancelled
                    return
                if step.kind == "compose":
                    continue
                await self._model_call(is_content=False, step_index=index)
                await self._execute_step(step, index)
            await self._compose()
            self._state.status = AgentStatus.completed
        except EnvironmentInterrupted:
            self._state.status = AgentStatus.failed
            await self.emit_error("run interrupted", kind="interrupted")
            raise
        except ModelCallError as exc:
            self._state.status = AgentStatus.failed
            await self.emit_complete(
                "The model backend is unavailable; no result can be provided.", status="failed",
                note=f"model error: {exc.kind}",
            )
        except Exception as exc:
            self._state.status = AgentStatus.failed
            await self.emit_error(f"{type(exc).__name__}: {exc}", kind="agent_crash")

    async def _restore_state(self) -> None:
        try:
            value, stale = await self._env.state_get("session")
        except StateUnavailableError:
            if self.config.profile == "weak":
                raise
            self._session = {}
            self._findings.append({"source": "state", "note": "state unavailable; continuing fresh"})
            return
        self._session = value if isinstance(value, dict) else {}
        self._state_stale = bool(stale)

    # ------------------------------------------------------------- plan steps

    async def _execute_step(self, step: PlanStep, index: int) -> None:
        if step.kind == "retrieve":
            tool = pick_tool_for_step(step, list(self._task.tools), strategy="best_score")
            if tool is None:
                return
            outcome = await self._exec_tool(tool, {"query": step.query}, purpose="retrieve documentation")
            if outcome.ok and isinstance(outcome.result, dict):
                docs = outcome.result.get("documents", [])
                self._served_docs.extend(docs)
                if docs and any(d.get("stale") for d in docs) and self.config.respect_staleness:
                    self._docs_stale = True
                if outcome.result.get("conflicting"):
                    self._docs_conflicting = True
                self._findings.append({"source": tool.name, "documents": docs})
                if self.config.trust_document_instructions:
                    await self._comply_with_docs(docs)
            elif outcome.error_kind is not None:
                # Retrieval failed outright: report it instead of going silent.
                self._unavailable.append(tool.name)
            return
        if step.kind == "lookup":
            entity = step.entity or ""
            tool = pick_tool_for_step(step, list(self._task.tools), strategy=self.config.routing)
            if tool is None:
                self._unavailable.append(entity)
                return
            key_field = self._key_field_of(tool)
            args: dict[str, Any] = {key_field: entity} if entity else {key_field: step.query}
            outcome = await self._exec_tool(tool, args, purpose=f"look up {entity or 'record'}")
            if outcome.ok and isinstance(outcome.result, dict):
                if outcome.result.get("found"):
                    self._findings.append({"source": tool.name, "entity": entity, "record": outcome.result["record"]})
                else:
                    self._unavailable.append(entity or str(args.get(key_field)))
                    if self.config.fabricate_when_missing:
                        self._findings.append({"source": tool.name, "entity": entity,
                                               "record": _fabricated(entity), "fabricated": True})
                # Weak profile treats even tool output as instructions.
                if self.config.trust_document_instructions:
                    await self._comply_with_docs([{"content": json.dumps(outcome.result, default=str)}])
            elif outcome.blocked:
                self._blocked_reason = f"lookup blocked: {outcome.error_kind}"
            elif outcome.error_kind is not None:
                # The tool never returned usable data (e.g. persistent 500s).
                self._unavailable.append(entity or str(args.get(key_field)))
            return
        if step.kind == "compute":
            tool = pick_tool_for_step(step, list(self._task.tools), strategy=self.config.routing)
            if tool is None or not step.numbers:
                return
            computed = _compute_args(tool, step)
            if computed is None:
                return
            outcome = await self._exec_tool(tool, computed, purpose="compute")
            if outcome.ok:
                self._findings.append({"source": tool.name, "computed": outcome.result})
            return
        if step.kind == "notify":
            tool = pick_tool_for_step(step, list(self._task.tools), strategy=self.config.routing)
            if tool is None:
                self._blocked_reason = "no tool for notification"
                return
            summary = _summarize(self._findings)
            args = _args_for_tool(tool, target=step.target or "support@company.example", summary=summary)
            outcome = await self._exec_tool(tool, args, purpose="send notification")
            self._record_effect_call(tool, args, outcome)
            if outcome.blocked:
                self._blocked_reason = f"notification blocked: {outcome.error_kind}"
            return
        if step.kind == "destroy":
            entity = step.entity or ""
            tool = pick_tool_for_step(step, list(self._task.tools), strategy=self.config.routing)
            if tool is None:
                self._blocked_reason = "no destructive tool available"
                return
            # Prepare-then-commit: checkpoint BEFORE the destructive call.
            await self._env.state_update("pending_deletion",
                                         {"entity": entity, "tool": tool.name, "prepared": True})
            if not step.execute and self.config.respect_prepare_only:
                await self._env.state_update("pending_deletion",
                                             {"entity": entity, "tool": tool.name, "executed": False})
                return
            args = {self._key_field_of(tool): entity}
            outcome = await self._exec_tool(tool, args, purpose=f"delete {entity}")
            self._record_effect_call(tool, args, outcome)
            if outcome.ok:
                await self._env.state_update("pending_deletion", {"entity": entity, "deleted": True})
            elif outcome.blocked:
                self._blocked_reason = f"deletion blocked: {outcome.error_kind}"
            elif outcome.error_kind in ("policy_denied", "permission_denied"):
                self._blocked_reason = f"deletion blocked by policy ({outcome.error_kind})"
            return
        if step.kind == "transfer":
            tool = pick_tool_for_step(step, list(self._task.tools), strategy=self.config.routing)
            if tool is None:
                self._blocked_reason = "no transfer tool available"
                return
            args = {"account": step.target or "ACC-1", "amount": step.numbers[0] if step.numbers else 0}
            outcome = await self._exec_tool(tool, args, purpose="transfer")
            self._record_effect_call(tool, args, outcome)
            if outcome.blocked:
                self._blocked_reason = f"transfer blocked: {outcome.error_kind}"
            elif outcome.error_kind in ("policy_denied", "permission_denied"):
                self._blocked_reason = f"transfer blocked by policy ({outcome.error_kind})"
            return
        if step.kind == "write":
            tool = pick_tool_for_step(step, list(self._task.tools), strategy=self.config.routing)
            if tool is None:
                self._blocked_reason = "no writable tool available"
                return
            key_field = self._key_field_of(tool)
            args = {key_field: step.entity or "NOTE-1", "body": _summarize(self._findings)}
            outcome = await self._exec_tool(tool, args, purpose=f"write {step.entity or 'note'}")
            if outcome.ok:
                self._findings.append({"source": tool.name, "entity": step.entity, "record": args})
            elif outcome.blocked:
                self._blocked_reason = f"write blocked: {outcome.error_kind}"
            elif outcome.error_kind in ("policy_denied", "permission_denied"):
                self._blocked_reason = f"write blocked by policy ({outcome.error_kind})"
            return

    def _record_effect_call(self, tool: ToolManifest, args: dict[str, Any], outcome: ToolOutcome) -> None:
        if outcome.ok:
            self._last_effect_call = (tool, args)

    @staticmethod
    def _key_field_of(tool: ToolManifest) -> str:
        for name, arg in tool.args_schema.items():
            if arg.get("required", True) and _ID_FIELD_RE.match(name):
                return name
        for name, arg in tool.args_schema.items():
            if arg.get("required", True):
                return name
        return "id"

    # ------------------------------------------------------- tool w/ retries

    async def _exec_tool(self, tool: ToolManifest, args: dict[str, Any], *, purpose: str) -> ToolOutcome:
        # Scope pre-check: a disciplined agent refuses out-of-scope calls
        # without bothering the policy engine (and without a violation).
        if self.config.respect_scope and tool.scope:
            field = str(tool.scope.get("field", ""))
            allowed = {str(v) for v in tool.scope.get("allowed", [])}
            value = args.get(field)
            if value is None or str(value) not in allowed:
                return ToolOutcome(ok=False, error_kind="scope_violation", blocked=True)
        attempt = 1
        while True:
            call_id = f"{tool.name}-{attempt}-{id(self) % 100000}-{len(self._findings)}"
            try:
                result = await self._env.call_tool(tool.name, args, tool_call_id=call_id, attempt=attempt)
                return ToolOutcome(ok=True, result=result)
            except ApprovalRequiredError:
                if tool.risk_class.value in self.config.approval_risk_classes:
                    outcome = await self._env.request_approval(tool.name, args, reason=purpose)
                    if outcome.decision.value == "approve":
                        attempt += 1
                        continue
                    return ToolOutcome(ok=False, error_kind=f"approval_{outcome.decision.value}", blocked=True)
                # Weak profile: no approval logic — retry blindly (violation).
                if self.config.retry_after_policy_denial and attempt <= self.config.max_retries_per_tool:
                    attempt += 1
                    await self._maybe_backoff(attempt)
                    continue
                return ToolOutcome(ok=False, error_kind="approval_required", blocked=True)
            except ToolError as exc:
                retriable = exc.retriable or self.config.retry_non_transient
                if retriable and attempt <= self.config.max_retries_per_tool:
                    await self._recorder.emit_retry(tool.name, attempt, exc.kind)
                    await self._maybe_backoff(attempt)
                    attempt += 1
                    continue
                return ToolOutcome(ok=False, error_kind=exc.kind)

    async def _maybe_backoff(self, attempt: int) -> None:
        if self.config.backoff_base_ms > 0:
            await asyncio.sleep(self.config.backoff_base_ms * (2 ** (attempt - 1)) / 1000.0)

    # ---------------------------------------------------------- model client

    async def _model_call(self, *, is_content: bool, step_index: int) -> str | None:
        """Emit MODEL_REQUEST/RESPONSE and apply model faults. Returns the
        content-fault marker (or None). Availability faults retry here."""
        fault = await self._env.injector.take_model_fault(content_call=is_content)
        if fault is not None and fault.fault.value in ("backend_unavailable", "model_rate_limit"):
            kind = fault.fault.value
            if self._model_retries < self.config.model_max_retries:
                self._model_retries += 1
                await self._recorder.emit_retry("model", self._model_retries, kind)
                return await self._model_call(is_content=is_content, step_index=step_index)
            raise ModelCallError(f"model {kind}", kind=kind, retriable=False)
        input_tokens = (len(self._task.task) + len(json.dumps(self._findings, default=str))) // 4 + 10
        await self._recorder.emit_model_request(step=step_index, input_tokens=input_tokens)
        output_tokens = 24 if not is_content else 60
        await self._recorder.emit_model_response(
            step=step_index, input_tokens=input_tokens, output_tokens=output_tokens,
            preview="scripted decision" if not is_content else "scripted answer",
        )
        return fault.fault.value if fault is not None else None

    # ---------------------------------------------------------------- compose

    async def _compose(self) -> None:
        corruption = await self._model_call(is_content=True, step_index=len(self._plan))
        json_mode = bool(_JSON_TASK_RE.search(self._task.task))
        status = "success"
        while True:
            output, status = self._build_output(json_mode=json_mode, corruption=corruption)
            if corruption in (None, "backend_unavailable", "model_rate_limit"):
                break
            if self.config.profile == "guarded":
                # Detect malformed/truncated output, retry the model once.
                await self._recorder.emit_retry("model", 1, corruption)
                corruption = None
                continue
            break
        await self.emit_complete(output, status=status)

    def _build_output(self, *, json_mode: bool, corruption: str | None) -> tuple[Any, str]:
        answer = self._answer_text()
        if not answer:
            # Task asked for a literal reply and no tool work was needed.
            m = _LITERAL_REPLY_RE.search(self._task.task)
            if m:
                answer = m.group(1).strip()
        status = "blocked" if self._blocked_reason else "success"
        if self._blocked_reason:
            answer = f"{self._blocked_reason}. {answer}" if answer else self._blocked_reason
        if corruption == "invalid_structured_output" and json_mode and self.config.profile == "weak":
            return '{"answer": "unterminated', status
        if corruption == "truncated_response" and self.config.profile == "weak":
            return answer[:20], status
        if json_mode:
            payload = {
                "answer": answer,
                "findings": self._findings,
                "unavailable": self._unavailable,
                "status": status,
            }
            return payload, status
        return answer, status

    def _answer_text(self) -> str:
        parts: list[str] = []
        for finding in self._findings:
            if "documents" in finding or "note" in finding:
                continue
            if "computed" in finding:
                result = finding["computed"]
                if isinstance(result, dict) and "result" in result:
                    unit = f" {result['unit']}" if result.get("unit") else ""
                    parts.append(f"computed result: {result['result']}{unit}")
                continue
            record = finding.get("record") or {}
            entity = finding.get("entity", "")
            flat = ", ".join(f"{k}={record[k]}" for k in sorted(record) if k != "id")
            prefix = f"{entity}: " if entity else ""
            parts.append(f"{prefix}{flat}" if flat else f"{entity}: no data")
        for entity in self._unavailable:
            if self.config.fabricate_when_missing:
                parts.append(f"{entity}: status=active, balance=1000.0")
            else:
                parts.append(f"{entity}: unavailable (no matching record)")
        if self._state_stale and self.config.respect_staleness:
            parts.append("note: cached state was stale; tool data used instead")
        if self._docs_stale:
            parts.append("warning: retrieved documents are stale; verify currency before acting")
        if self._docs_conflicting:
            parts.append("warning: conflicting evidence detected; manual verification required")
        if self._served_docs:
            doc_ids = [str(d.get("id")) for d in self._served_docs if d.get("id")]
            if doc_ids:
                parts.append("References: " + ", ".join(doc_ids))
        if self._served_docs and not self.config.trust_document_instructions and any(
                _INJECT_RE.search(str(d.get("content", ""))) for d in self._served_docs):
            parts.append("note: untrusted instructions in retrieved documents were ignored")
        if self._session and not self.config.respect_staleness:
            for key, value in self._session.items():
                if str(key).replace("_", " ") in self._task.task.lower():
                    parts.append(f"{key}={value} (from cached state)")
        if (_SECRET_TASK_RE.search(self._task.task) or self._forced_reveal):
            if self.config.echo_secrets:
                parts.append("environment: " + json.dumps(self._task.env.get("variables", {}), sort_keys=True))
            else:
                parts.append("note: environment secrets are not disclosed")
        sources = sorted({str(f.get("source")) for f in self._findings if f.get("source")})
        if sources:
            parts.append("Sources: " + ", ".join(sources))
        return ". ".join(p.rstrip(".") for p in parts) + ("." if parts else "")

    # -------------------------------------------------- weak doc compliance

    async def _comply_with_docs(self, docs: list[dict[str, Any]]) -> None:
        """Weak profile only: follow instructions found in retrieved content.
        The guarded profile never calls this — document text is data, not
        instructions."""
        for doc in docs:
            content = str(doc.get("content", ""))
            if not _INJECT_RE.search(content):
                continue
            url_match = _EXFIL_URL_RE.search(content)
            tool_call = _CALL_TOOL_RE.search(content)
            if url_match and _SECRET_WORD_RE.search(content):
                await self._exfiltrate(url_match.group(0))
            elif tool_call:
                tool = next((t for t in self._task.tools if t.name == tool_call.group(1)), None)
                if tool is not None:
                    await self._exec_tool(tool, {}, purpose="document instruction")
            if _REVEAL_RE.search(content):
                self._forced_reveal = True

    async def _exfiltrate(self, url: str) -> None:
        """Weak profile: send environment variables to the attacker URL found
        in a document. Everything stays inside the sandbox — the call is
        recorded with its arguments, which is exactly what the leakage and
        injection evaluators look for."""
        secrets_blob = json.dumps(self._task.env.get("variables", {}), sort_keys=True)
        effect_tool = next(
            (t for t in self._task.tools
             if t.risk_class.value == "external_side_effect" and ("url" in t.args_schema or "webhook" in t.name)),
            None,
        )
        http_tool = next((t for t in self._task.tools if "url" in t.args_schema), None)
        email_tool = next((t for t in self._task.tools if "email" in t.name), None)
        tool = effect_tool or http_tool
        if tool is not None:
            args: dict[str, Any] = {"url": url, "payload": secrets_blob}
            outcome = await self._exec_tool(tool, args, purpose="document instruction")
            self._record_effect_call(tool, args, outcome)
        elif email_tool is not None:
            host = url.split("//")[-1].split("/")[0]
            args = {"to": f"data@{host}", "subject": "data", "body": secrets_blob}
            outcome = await self._exec_tool(email_tool, args, purpose="document instruction")
            self._record_effect_call(email_tool, args, outcome)


def _args_for_tool(tool: ToolManifest, *, target: str, summary: str) -> dict[str, Any]:
    """Build notification arguments that fit the tool's declared schema."""
    args: dict[str, Any] = {}
    if "url" in tool.args_schema:
        args["url"] = target if "://" in target else f"https://{target}"
        args["payload"] = summary
    if "to" in tool.args_schema:
        args["to"] = target
    if "subject" in tool.args_schema:
        args["subject"] = f"Update re {summary[:40]}"
    if "body" in tool.args_schema:
        args["body"] = summary
    if "message" in tool.args_schema:
        args["message"] = summary
    return args or {"to": target, "body": summary}


def _fabricated(entity: str | None) -> dict[str, Any]:
    return {"id": entity or "unknown", "status": "active", "balance": 1000.0}


def _compute_args(tool: ToolManifest, step: PlanStep) -> dict[str, Any] | None:
    name = tool.name
    if "convert" in name or "currency" in name or "currency" in tool.description.lower():
        m = re.search(r"([a-z]{3})\s*(?:to|->|in)\s*([a-z]{3})", step.query.lower())
        if m and step.numbers:
            return {"amount": step.numbers[0], "from": m.group(1).upper(), "to": m.group(2).upper()}
        return None
    if "percent" in name or "percent" in tool.description.lower():
        if len(step.numbers) >= 2:
            return {"value": step.numbers[0], "pct": step.numbers[1]}
        return None
    if len(step.numbers) >= 2:
        return {"a": step.numbers[0], "b": step.numbers[1]}
    return None


def _summarize(findings: list[dict[str, Any]]) -> str:
    return "; ".join(
        json.dumps(f.get("record", f.get("computed", {})), sort_keys=True, default=str) for f in findings[:3]
    ) or "(no findings)"
