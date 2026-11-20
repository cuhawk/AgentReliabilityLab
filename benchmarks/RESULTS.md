# Wave 4 benchmark results

Run `run-20261120-063124-daf3b986` evaluated the guarded reference agent offline on
Python 3.12.14 with three trials per scenario across all 84
scenarios in 16 categories and concurrency capped at four.

```bash
uv run arl eval suites/full.yaml --agent reference:guarded --trials 3 \
  --k 1 --k 2 --k 3 --concurrency 4 --seed 42 --output-dir .arl/wave4 --db .arl/wave4/runs.sqlite
```

| Metric | Observed |
|---|---:|
| Passed trials | 252/252 |
| Trial pass rate | 100% |
| Trial pass rate, Wilson 95% CI | [98.50%, 100.00%] |
| p50 trial latency | 0.510 ms |
| p95 trial latency | 32.911 ms |

| k | pass@k | pass^k | Scenarios |
|---:|---:|---:|---:|
| 1 | 100% | 100% | 84 |
| 2 | 100% | 100% | 84 |
| 3 | 100% | 100% | 84 |

## Cost vs. reliability Pareto frontier

| Scenario | Trials | Pass rate | Mean tokens | p95 latency (ms) |
|---|---:|---:|---:|---:|
| `delegation-self-cycle` | 3 | 100% | 0.00 | 0.211 |

The frontier compares scenario trial groups within this run. Each row has no
other scenario with an equal or higher pass rate, equal or lower mean total
tokens, and equal or lower p95 latency with at least one strict improvement.
Token totals include input and output, and costs include all trials. A trial
passes if it meets its scenario expectations, including an expected block.

Latency measures scenario execution and excludes scheduler queue time. The Wilson
interval pools trial outcomes. Repeated deterministic trials share fixtures, so
interpret the interval with that dependence in mind.
