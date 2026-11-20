# Wave 3 benchmark results

Run `run-20261120-052900-7b870f58` ran the full offline suite with the guarded reference
agent on Python 3.12.14. It ran three trials for each
of 76 scenarios across 15 categories, with concurrency capped at four.

```bash
uv run arl eval suites/full.yaml --agent reference:guarded --trials 3 \
  --k 1 --k 2 --k 3 --concurrency 4 --seed 42 --output-dir .arl/wave3 --db .arl/wave3/runs.sqlite
```

| Metric | Observed |
|---|---:|
| Passed trials | 228/228 |
| Trial pass rate | 100% |
| Trial pass rate Wilson 95% interval | [98.34%, 100.00%] |
| p50 trial latency | 0.553 ms |
| p95 trial latency | 33.137 ms |

| k | pass@k | pass^k | Scenarios |
|---:|---:|---:|---:|
| 1 | 100% | 100% | 76 |
| 2 | 100% | 100% | 76 |
| 3 | 100% | 100% | 76 |

Latency measures scenario execution and excludes scheduler queue time. The Wilson
interval describes pooled trial outcomes. Repeated deterministic trials share
fixtures, so the interval assumes more independence than this run provides.
