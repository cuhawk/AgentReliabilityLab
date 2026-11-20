import asyncio

import httpx
import pytest

from arl.evaluation.aggregates import aggregate_consistency
from arl.evaluation.engine import ScenarioResult
from arl.scheduler import EvaluationScheduler


def test_consistency_averages_scenarios_over_trial_subsets() -> None:
    results = [
        ScenarioResult(scenario_id=scenario, trial=index, status=status)
        for scenario, statuses in [("mixed", ["pass", "pass", "fail"]), ("failed", ["fail"] * 3)]
        for index, status in enumerate(statuses, 1)
    ]
    metrics = aggregate_consistency(results, [1, 2, 3])
    assert metrics["pass@1"].value == pytest.approx(1 / 3)
    assert metrics["pass^1"].value == pytest.approx(1 / 3)
    assert metrics["pass@2"].value == pytest.approx(1 / 2)
    assert metrics["pass^2"].value == pytest.approx(1 / 6)
    assert metrics["pass@3"].value == pytest.approx(1 / 2)
    assert metrics["pass^3"].value == 0
    assert metrics["pass^3"].n == 2
    with pytest.raises(ValueError, match="number of trials"):
        aggregate_consistency(results, [4])


async def test_retry_budget_is_shared_and_jitter_is_seeded() -> None:
    scheduler = EvaluationScheduler(concurrency=4, retry_budget=1, seed=42)
    other = EvaluationScheduler(concurrency=4, retry_budget=0, seed=42)
    assert scheduler.backoff("one", 1) == other.backoff("one", 1)
    assert scheduler.backoff("one", 1) != other.backoff("two", 1)
    request = httpx.Request("POST", "https://example.test/turn")
    calls = 0

    async def throttled() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, request=request)

    outcomes = await asyncio.gather(
        scheduler.request(throttled, key="one"),
        scheduler.request(throttled, key="two"),
        return_exceptions=True,
    )
    assert calls == 3
    assert all(isinstance(outcome, httpx.HTTPStatusError) for outcome in outcomes)


async def test_scheduler_caps_in_flight_work() -> None:
    scheduler = EvaluationScheduler(concurrency=2, retry_budget=0, seed=0)
    active = peak = 0
    both_started = asyncio.Event()

    async def work() -> None:
        nonlocal active, peak
        async with scheduler.slot():
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_started.set()
            await both_started.wait()
            active -= 1

    await asyncio.wait_for(asyncio.gather(*(work() for _ in range(5))), timeout=2)
    assert peak == 2
