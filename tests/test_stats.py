import pytest

from src.metrics.stats import cluster_bootstrap_diff, cluster_bootstrap_rate, holm


def rows(rates: dict[str, list[int]]) -> list[dict]:
    return [{"cluster": c, "y": y} for c, ys in rates.items() for y in ys]


def test_identical_arms_give_a_zero_difference_and_an_interval_around_it():
    a = rows({"t1": [1, 0], "t2": [0, 0], "t3": [1, 1]})
    result = cluster_bootstrap_diff(a, a, "cluster", "y", n_resamples=2000, seed=0)
    assert result["diff"] == 0
    assert result["ci_low"] <= 0 <= result["ci_high"]
    assert result["p_two_sided"] == pytest.approx(1.0)


def test_a_large_consistent_gap_excludes_zero():
    a = rows({f"t{i}": [1, 1, 1, 1, 1] for i in range(10)})
    b = rows({f"t{i}": [0, 0, 0, 0, 1] for i in range(10)})
    result = cluster_bootstrap_diff(a, b, "cluster", "y", n_resamples=2000, seed=0)
    assert result["diff"] == pytest.approx(0.8)
    assert result["ci_low"] > 0
    assert result["p_two_sided"] < 0.01


def test_resampling_is_by_cluster_shared_across_arms():
    # one task carries the whole effect, so the interval must be wide and include zero
    a = rows({"t1": [1] * 5, **{f"t{i}": [0] * 5 for i in range(2, 11)}})
    b = rows({f"t{i}": [0] * 5 for i in range(1, 11)})
    result = cluster_bootstrap_diff(a, b, "cluster", "y", n_resamples=4000, seed=0)
    assert result["ci_low"] == 0


def test_holm_adjusts_in_step_down_order_and_stays_monotone():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]


def test_rate_interval_contains_the_rate_and_is_zero_width_when_every_cluster_agrees():
    mixed = cluster_bootstrap_rate(rows({"t1": [1, 0], "t2": [0, 0], "t3": [1, 1]}), "cluster", "y", 2000, 0)
    assert mixed["rate"] == pytest.approx(0.5) and mixed["ci_low"] <= 0.5 <= mixed["ci_high"]
    flat = cluster_bootstrap_rate(rows({"t1": [0, 0], "t2": [0, 0]}), "cluster", "y", 2000, 0)
    assert (flat["ci_low"], flat["ci_high"]) == (0.0, 0.0)
