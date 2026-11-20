from arl.evaluation.aggregates import pareto_frontier
from arl.evaluation.engine import ScenarioResult


def test_pareto_frontier_keeps_tradeoffs_and_ties() -> None:
    def trial(name: str, tokens: int, duration: float, status: str = "pass") -> ScenarioResult:
        return ScenarioResult(
            scenario_id=name, tokens_in=tokens - 1, tokens_out=1, duration_ms=duration, status=status,
        )

    results = [
        trial("cheap", 5, 1), trial("cheap", 5, 3, "fail"),
        trial("cheap-tie", 5, 1), trial("cheap-tie", 5, 3, "fail"),
        trial("fast", 10, 1),
        trial("reliable", 8, 3),
        trial("dominated", 10, 1, "fail"),
    ]
    points = pareto_frontier(list(reversed(results)))
    assert [point.scenario_id for point in points] == ["reliable", "fast", "cheap", "cheap-tie"]
    assert points[2].trials == 2
    assert points[2].pass_rate == 0.5
    assert points[2].mean_tokens == 5
    assert points[2].p95_latency_ms == 2.9
    assert pareto_frontier([]) == []
