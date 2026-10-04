"""Report rendering: JSON (machine), Markdown (human/PR), HTML (self-contained).
Reports include environment, agent version, model, suite, timestamp, metrics
with confidence intervals, failed scenarios with trace links, and — when a
baseline was given — the comparison table."""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

REPORT_VERSION = 1


def build_report_dict(run: BaseModel, comparison: dict[str, Any] | None = None) -> dict[str, Any]:
    results = run.results  # type: ignore[attr-defined]
    metrics = run.metrics  # type: ignore[attr-defined]
    failed = [r for r in results if r.status != "pass"]
    return {
        "report_version": REPORT_VERSION,
        "run_id": run.run_id,  # type: ignore[attr-defined]
        "suite": run.suite_id,  # type: ignore[attr-defined]
        "agent_id": run.agent_id,  # type: ignore[attr-defined]
        "agent_version": run.config.get("agent_spec"),  # type: ignore[attr-defined]
        "model": run.model,  # type: ignore[attr-defined]
        "created_at": run.created_at,  # type: ignore[attr-defined]
        "environment": run.environment,  # type: ignore[attr-defined]
        "config": run.config,  # type: ignore[attr-defined]
        "summary": {"total": len(results), "passed": run.passed, "failed": run.failed},  # type: ignore[attr-defined]
        "metrics": {name: m.model_dump() for name, m in metrics.items()},
        "scenarios": [
            {
                "scenario_id": r.scenario_id,
                "title": r.title,
                "category": r.category,
                "status": r.status,
                "final_status": r.final_status,
                "trace_status": r.trace_status,
                "duration_ms": r.duration_ms,
                "faults": r.faults,
                "tool_calls": r.tool_calls,
                "retries": r.retries,
                "violations": [v.model_dump() for v in r.violations],
                "trace_link": f"traces/{r.scenario_id}.trace.txt",
            }
            for r in results
        ],
        "failed_scenarios": [
            {
                "scenario_id": r.scenario_id,
                "category": r.category,
                "failed_checks": [
                    {
                        "evaluator": c.evaluator,
                        "metric": c.metric,
                        "expected": c.expected,
                        "observed": c.observed,
                        "message": c.message,
                    }
                    for c in r.failed_checks
                ],
                "trace_link": f"traces/{r.scenario_id}.trace.txt",
            }
            for r in failed
        ],
        "baseline_comparison": comparison,
    }


def write_reports(run: BaseModel, output_dir: Path, *, formats: list[str],
                  comparison: dict[str, Any] | None = None) -> dict[str, str]:
    run_dir = Path(output_dir) / str(run.run_id)  # type: ignore[attr-defined]
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    prime_trace_cache(run)
    for result in run.results:  # type: ignore[attr-defined]
        (traces_dir / f"{result.scenario_id}.trace.txt").write_text(result.trace_text, encoding="utf-8")
        if result.trace is not None:
            (traces_dir / f"{result.scenario_id}.trace.json").write_text(
                json.dumps(result.trace.model_dump(), indent=2, default=str), encoding="utf-8"
            )

    report = build_report_dict(run, comparison)
    paths: dict[str, str] = {}
    if "json" in formats:
        path = run_dir / "report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        paths["json"] = str(path)
    if "md" in formats:
        path = run_dir / "report.md"
        path.write_text(render_markdown(report), encoding="utf-8")
        paths["md"] = str(path)
    if "html" in formats:
        path = run_dir / "report.html"
        path.write_text(render_html(report), encoding="utf-8")
        paths["html"] = str(path)
    return paths


# --------------------------------------------------------------------------- markdown


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# AgentReliabilityLab report — `{report['run_id']}`")
    lines.append("")
    lines.append(f"* Suite: **{report['suite']}**")
    lines.append(f"* Agent: **{report['agent_id']}** (version spec: `{report.get('agent_version')}`)")
    lines.append(f"* Model: **{report['model']}**")
    lines.append(f"* Finished: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(report['created_at']))} UTC")
    lines.append(f"* Environment: Python {report['environment'].get('python')}, "
                 f"ARL {report['environment'].get('arl_version')}, {report['environment'].get('platform')}")
    lines.append("")
    s = report["summary"]
    lines.append(f"**{s['passed']}/{s['total']} scenarios passed.**")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value | 95% CI | n | Source |")
    lines.append("|---|---|---|---|---|")
    for name, m in report["metrics"].items():
        value = f"{m['value']:.4f}" if isinstance(m["value"], float) else str(m["value"])
        ci = ""
        if m.get("ci_low") is not None:
            ci = f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
        lines.append(f"| {name} | {value} | {ci} | {m.get('n') or ''} | {m.get('source', '')} |")
    lines.append("")

    if report.get("baseline_comparison"):
        comp = report["baseline_comparison"]
        lines.append(f"## Baseline comparison — verdict: **{comp['verdict']}**")
        lines.append("")
        lines.append("```")
        for row in comp.get("rows", []):
            base = "n/a" if row.get("baseline") is None else f"{row['baseline']:.4f}"
            cand = "n/a" if row.get("candidate") is None else f"{row['candidate']:.4f}"
            delta = "" if row.get("delta") is None else f"{row['delta']:+.4f}"
            lines.append(f"{row['metric']:34} {base:>10} {cand:>10} {delta:>10}  {row['status']}")
        lines.append("```")
        lines.append("")

    if report["failed_scenarios"]:
        lines.append("## Failed scenarios")
        for f in report["failed_scenarios"]:
            lines.append(f"### `{f['scenario_id']}` ({f['category']})")
            for c in f["failed_checks"]:
                lines.append(f"* **{c['metric']}** — expected {c['expected']!r}, observed {c['observed']!r}")
                if c.get("message"):
                    lines.append(f"  * {c['message']}")
            lines.append(f"  * trace: `{f['trace_link']}`")
        lines.append("")

    lines.append("## All scenarios")
    lines.append("")
    lines.append("| Scenario | Category | Status | Final | Faults | Tool calls | Retries | Trace |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sc in report["scenarios"]:
        faults = ", ".join(sc["faults"]) or "—"
        lines.append(
            f"| `{sc['scenario_id']}` | {sc['category']} | {sc['status']} | {sc['final_status']} "
            f"| {faults} | {sc['tool_calls']} | {sc['retries']} | `{sc['trace_link']}` |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- html


_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 0; background: #0f1117; color: #e6e6e6; }
main { max-width: 1100px; margin: 0 auto; padding: 24px; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #2a2f3a; padding-bottom: 6px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.92rem; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #23272f; }
th { color: #9aa3b2; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.06em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: #4cc38a; } .fail { color: #ff6b6b; font-weight: 600; }
code, pre { font-family: ui-monospace, 'Cascadia Code', Consolas, monospace; }
pre { background: #161a22; border: 1px solid #23272f; padding: 12px; border-radius: 8px;
overflow-x: auto; font-size: 0.8rem; }
.meta { color: #9aa3b2; font-size: 0.9rem; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.badge.pass { background: rgba(76,195,138,.15); } .badge.fail { background: rgba(255,107,107,.15); }
"""


def render_html(report: dict[str, Any]) -> str:
    e = html.escape
    s = report["summary"]
    verdict_badge = "pass" if s["failed"] == 0 else "fail"
    rows_html = []
    for name, m in report["metrics"].items():
        ci = ""
        if m.get("ci_low") is not None:
            ci = f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
        rows_html.append(
            f"<tr><td>{e(name)}</td><td class='num'>{m['value']:.4f}</td>"
            f"<td class='num'>{ci}</td><td class='num'>{m.get('n') or ''}</td>"
            f"<td>{e(str(m.get('source', '')))}</td></tr>"
        )

    scenario_rows = []
    for sc in report["scenarios"]:
        cls = "pass" if sc["status"] == "pass" else "fail"
        scenario_rows.append(
            f"<tr><td><code>{e(sc['scenario_id'])}</code></td><td>{e(sc['category'])}</td>"
            f"<td class='{cls}'>{sc['status']}</td><td>{e(sc['final_status'])}</td>"
            f"<td>{e(', '.join(sc['faults']) or '—')}</td><td class='num'>{sc['tool_calls']}</td>"
            f"<td class='num'>{sc['retries']}</td><td><code>{e(sc['trace_link'])}</code></td></tr>"
        )

    failures_html = []
    for f in report["failed_scenarios"]:
        checks = "".join(
            f"<li><strong>{e(c['metric'])}</strong> — expected <code>{e(str(c['expected']))}</code>, "
            f"observed <code>{e(str(c['observed']))}</code>"
            f"{'<br><em>' + e(c['message']) + '</em>' if c.get('message') else ''}</li>"
            for c in f["failed_checks"]
        )
        failures_html.append(f"<h3><code>{e(f['scenario_id'])}</code></h3><ul>{checks}</ul>")

    comparison_html = ""
    if report.get("baseline_comparison"):
        comp = report["baseline_comparison"]
        comp_rows = "".join(
            f"<tr><td>{e(r['metric'])}</td>"
            f"<td class='num'>{'' if r.get('baseline') is None else format(r['baseline'], '.4f')}</td>"
            f"<td class='num'>{'' if r.get('candidate') is None else format(r['candidate'], '.4f')}</td>"
            f"<td class='num'>{'' if r.get('delta') is None else format(r['delta'], '+.4f')}</td>"
            f"<td class=\"{'pass' if r['status'].startswith('pass') else 'fail'}\">{e(r['status'])}</td>"
            f"<td>{e(str(r.get('note', '')))}</td></tr>"
            for r in comp.get("rows", [])
        )
        comparison_html = (
            f"<h2>Baseline comparison — <span class=\"badge "
            f"{'pass' if comp['verdict'] == 'pass' else 'fail'}\">{e(comp['verdict'])}</span></h2>"
            f"<table><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th>"
            f"<th>Gate</th><th>Note</th></tr>{comp_rows}</table>"
        )

    traces_html = []
    for sc in report["scenarios"]:
        if sc["status"] != "pass":
            trace_text = _trace_text_cache.get(sc["scenario_id"], "")
            traces_html.append(f"<h3><code>{e(sc['scenario_id'])}</code></h3><pre>{e(trace_text)}</pre>")

    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(report["created_at"]))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>ARL report {e(report['run_id'])}</title><style>{_CSS}</style></head>
<body><main>
<h1>AgentReliabilityLab report <span class="badge {verdict_badge}">{s['passed']}/{s['total']} passed</span></h1>
<p class="meta">suite <code>{e(report['suite'])}</code> · agent <code>{e(report['agent_id'])}</code>
· model <code>{e(report['model'])}</code> · {stamp}
· ARL {e(str(report['environment'].get('arl_version')))} · Python {e(str(report['environment'].get('python')))}</p>
<h2>Metrics</h2>
<table><tr><th>Metric</th><th>Value</th><th>95% CI</th><th>n</th><th>Source</th></tr>{''.join(rows_html)}</table>
{comparison_html}
<h2>Failed scenarios</h2>
{''.join(failures_html) if failures_html else '<p class="pass">None — all scenarios passed.</p>'}
<h2>All scenarios</h2>
<table><tr><th>Scenario</th><th>Category</th><th>Status</th><th>Final</th><th>Faults</th>
<th>Tools</th><th>Retries</th><th>Trace</th></tr>{''.join(scenario_rows)}</table>
<h2>Failure traces</h2>
{''.join(traces_html)}
</main></body></html>"""


_trace_text_cache: dict[str, str] = {}


def prime_trace_cache(run: BaseModel) -> None:
    """Called by write_reports so failed traces can be embedded in HTML."""
    _trace_text_cache.clear()
    for r in run.results:  # type: ignore[attr-defined]
        _trace_text_cache[r.scenario_id] = r.trace_text
