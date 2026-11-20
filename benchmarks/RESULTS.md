# Wave 5 offline benchmark results

The guarded run `run-20261120-081138-7ef1bd7f` and weak run
`run-20261120-081139-ddbcd690` evaluated all 92 scenarios in 17 categories on
Python 3.12.14. Latency measures scenario execution and excludes scheduler queue
time.

```bash
uv run arl eval suites/full.yaml --agent reference:guarded --trials 3 \
  --k 1 --k 2 --k 3 --concurrency 4 --seed 42 --output-dir .arl/wave5 --formats json
uv run arl eval suites/full.yaml --agent reference:weak --concurrency 4 \
  --seed 42 --output-dir .arl/wave5-weak --formats json --no-fail
```

| Agent | Passed trials | Trial pass rate | Wilson 95% interval | p50 trial latency | p95 trial latency |
|---|---:|---:|---:|---:|---:|
| Guarded | 276/276 | 100% | [98.63%, 100%] | 0.644 ms | 34.763 ms |
| Weak | 44/92 | 47.83% | [37.91%, 57.91%] | 0.745 ms | 9.002 ms |

The guarded agent passed all 92 scenarios in each of the three trials, including
24/24 trials across the eight verifier-integrity scenarios. Both `pass@k` and
`pass^k` were 100% for `k = 1, 2, 3`. Repeated deterministic trials share
fixtures, which limits the interpretation of the pooled Wilson interval.

## Offline calibration exercise

The exercise used observed weak-agent outcomes. A scripted judge passed a
sample when its reported final status matched the scenario's expected final
status. Full deterministic scenario verdicts served as reference labels. The
46 even-indexed scenarios sorted by ID formed the calibration set, and the 46
odd-indexed scenarios formed the held-out batch. Human-labeled LLM judge
agreement remains unmeasured.

| Metric | Result |
|---|---:|
| Calibration count (TP / TN / FP / FN) | 25 / 2 / 19 / 0 |
| Cohen’s κ | 0.1027 |
| TPR | 100% |
| TNR | 9.52% |
| Held-out scripted judge pass rate | 41/46 (89.13%) |
| Held-out deterministic pass rate | 19/46 (41.30%) |
| Held-out Rogan–Gladen estimate | 0% (clipped) |
| Held-out Rogan–Gladen 95% interval | [0%, 100%] |

The interval spans the full range because the calibration uncertainty allows
`TPR + TNR − 1` to cross zero.

```python
import json
from pathlib import Path

from arl.evaluation.calibration import calibrate, rogan_gladen
from arl.scenarios.suite import load_suite

path = Path(".arl/wave5-weak/run-20261120-081139-ddbcd690/report.json")
rows = sorted(json.loads(path.read_text())["scenarios"], key=lambda row: row["scenario_id"])
_, scenarios = load_suite("suites/full.yaml")
expected = {scenario.id: scenario.expect.final_status for scenario in scenarios}

def judge(row):
    return row["final_status"] == expected[row["scenario_id"]]

calibration = calibrate(
    gold=[row["status"] == "pass" for row in rows[::2]],
    judged=[judge(row) for row in rows[::2]],
)
held_out = rows[1::2]
print(calibration)
print(rogan_gladen(calibration, judged_passes=sum(map(judge, held_out)), total=len(held_out)))
```
