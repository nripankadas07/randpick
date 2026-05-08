"""Vose's alias-method core: build correctness + pick contract."""

from __future__ import annotations

import math

import pytest

from randpick._alias import alias_pick, build_alias_table


def test_build_alias_uniform_two_buckets_yields_unit_probabilities() -> None:
    prob, alias = build_alias_table([1.0, 1.0])
    assert prob == (1.0, 1.0)
    assert alias == (0, 0)


def test_build_alias_singleton_yields_unit_probability() -> None:
    prob, alias = build_alias_table([3.5])
    assert prob == (1.0,)
    assert alias == (0,)


def test_build_alias_skewed_two_buckets_redirects_small_to_large() -> None:
    prob, alias = build_alias_table([1.0, 9.0])
    assert prob[0] == pytest.approx(0.2)
    assert alias[0] == 1
    assert prob[1] == 1.0


def test_build_alias_three_buckets_distribution_matches_long_run() -> None:
    import random

    weights = [0.5, 0.2, 0.3]
    prob, alias = build_alias_table(weights)
    rng = random.Random(0xC0FFEE)
    counts = [0, 0, 0]
    trials = 60_000
    for _ in range(trials):
        column = rng.randrange(3)
        coin = rng.random()
        counts[alias_pick(prob, alias, column, coin)] += 1
    for index, weight in enumerate(weights):
        assert math.isclose(counts[index] / trials, weight, abs_tol=0.01)


def test_alias_pick_falls_through_to_alias_when_coin_above_prob() -> None:
    assert alias_pick((0.2, 1.0), (1, 0), column=0, coin=0.9) == 1
    assert alias_pick((0.2, 1.0), (1, 0), column=0, coin=0.0) == 0


def test_alias_pick_returns_column_when_prob_is_one() -> None:
    assert alias_pick((1.0, 1.0), (0, 0), column=1, coin=0.5) == 1


def test_build_alias_zero_weight_in_population_never_picked() -> None:
    import random

    prob, alias = build_alias_table([0.0, 1.0, 1.0])
    rng = random.Random(0xC0FFEE)
    counts = [0, 0, 0]
    for _ in range(20_000):
        column = rng.randrange(3)
        coin = rng.random()
        counts[alias_pick(prob, alias, column, coin)] += 1
    assert counts[0] == 0


def test_build_alias_many_buckets_returns_correct_table_shape() -> None:
    weights = [float(i + 1) for i in range(20)]
    prob, alias = build_alias_table(weights)
    assert len(prob) == 20
    assert len(alias) == 20
    for value in prob:
        assert 0.0 <= value <= 1.0
    for index in alias:
        assert 0 <= index < 20


def test_build_alias_resilient_to_residual_floating_point_pile_up() -> None:
    weights = [1.0] * 1000 + [1e-9]
    prob, alias = build_alias_table(weights)
    assert len(prob) == 1001
    assert len(alias) == 1001
    for value in prob:
        assert 0.0 <= value <= 1.0
