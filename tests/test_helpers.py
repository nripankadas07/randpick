"""Top-level helpers — weighted_choice, weighted_choices, weighted_sample, cumulative_pick."""

from __future__ import annotations

import math
import random

import pytest

import randpick


def test_weighted_choice_returns_only_member_for_singleton(
    rng: random.Random,
) -> None:
    assert randpick.weighted_choice(["x"], [1.0], rng=rng) == "x"


def test_weighted_choice_distribution_matches_weights(rng: random.Random) -> None:
    counts = {"a": 0, "b": 0}
    trials = 50_000
    for _ in range(trials):
        counts[randpick.weighted_choice(["a", "b"], [1.0, 4.0], rng=rng)] += 1
    assert math.isclose(counts["a"] / trials, 0.2, abs_tol=0.01)
    assert math.isclose(counts["b"] / trials, 0.8, abs_tol=0.01)


def test_weighted_choice_zero_weight_item_is_never_picked(
    rng: random.Random,
) -> None:
    drawn = {
        randpick.weighted_choice(["a", "skip", "b"], [1.0, 0.0, 1.0], rng=rng)
        for _ in range(2000)
    }
    assert drawn == {"a", "b"}


def test_weighted_choice_default_rng_uses_module_random() -> None:
    out = randpick.weighted_choice(["x"], [1.0])
    assert out == "x"


def test_weighted_choice_floating_point_edge_clamps_to_last_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRng:
        def random(self) -> float:
            return 1.0

    items = ("a", "b", "c")
    out = randpick.weighted_choice(items, [1.0, 1.0, 1.0], rng=FakeRng())  # type: ignore[arg-type]
    assert out == "c"


def test_weighted_choices_returns_k_items(rng: random.Random) -> None:
    out = randpick.weighted_choices(["a", "b", "c"], [1, 2, 3], 10, rng=rng)
    assert len(out) == 10
    assert all(item in {"a", "b", "c"} for item in out)


def test_weighted_choices_zero_k_returns_empty(rng: random.Random) -> None:
    assert randpick.weighted_choices(["a", "b"], [1, 1], 0, rng=rng) == []


def test_weighted_choices_distribution_matches_weights(rng: random.Random) -> None:
    out = randpick.weighted_choices(["a", "b"], [1, 9], 50_000, rng=rng)
    proportion_b = out.count("b") / len(out)
    assert math.isclose(proportion_b, 0.9, abs_tol=0.01)


def test_weighted_choices_clamps_floating_point_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRng:
        def random(self) -> float:
            return 1.0

    out = randpick.weighted_choices(
        ("a", "b", "c"),
        [1.0, 1.0, 1.0],
        3,
        rng=FakeRng(),  # type: ignore[arg-type]
    )
    assert all(item == "c" for item in out)


def test_weighted_sample_with_zero_k_returns_empty_list(rng: random.Random) -> None:
    assert randpick.weighted_sample(["a"], [1.0], 0, rng=rng) == []


def test_weighted_sample_returns_k_distinct_items(rng: random.Random) -> None:
    out = randpick.weighted_sample(
        ["a", "b", "c", "d", "e"],
        [1, 2, 3, 4, 5],
        3,
        rng=rng,
    )
    assert len(out) == 3
    assert len(set(out)) == 3


def test_weighted_sample_full_population_returns_all_items(rng: random.Random) -> None:
    items = ["a", "b", "c"]
    out = randpick.weighted_sample(items, [1, 1, 1], 3, rng=rng)
    assert sorted(out) == sorted(items)


def test_weighted_sample_zero_weight_items_excluded_when_possible(
    rng: random.Random,
) -> None:
    out = randpick.weighted_sample(
        ["a", "skip", "b", "c"],
        [1.0, 0.0, 1.0, 1.0],
        3,
        rng=rng,
    )
    assert "skip" not in out


def test_weighted_sample_seeded_rng_is_deterministic() -> None:
    a = randpick.weighted_sample(
        ["a", "b", "c", "d"],
        [1, 2, 3, 4],
        2,
        rng=random.Random(42),
    )
    b = randpick.weighted_sample(
        ["a", "b", "c", "d"],
        [1, 2, 3, 4],
        2,
        rng=random.Random(42),
    )
    assert a == b


def test_weighted_sample_high_weight_items_appear_more_often() -> None:
    rng = random.Random(0xC0FFEE)
    counts = {"a": 0, "b": 0}
    for _ in range(2000):
        out = randpick.weighted_sample(["a", "b"], [1.0, 9.0], 1, rng=rng)
        counts[out[0]] += 1
    assert counts["b"] > counts["a"] * 5


def test_cumulative_pick_uses_precomputed_cum(rng: random.Random) -> None:
    items = ("a", "b", "c")
    cum = [1.0, 3.0, 6.0]
    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(60_000):
        counts[randpick.cumulative_pick(cum, items, rng=rng)] += 1
    # weights are 1, 2, 3 → 1/6, 2/6, 3/6
    assert math.isclose(counts["a"] / 60_000, 1 / 6, abs_tol=0.01)
    assert math.isclose(counts["b"] / 60_000, 2 / 6, abs_tol=0.01)
    assert math.isclose(counts["c"] / 60_000, 3 / 6, abs_tol=0.01)


def test_cumulative_pick_empty_items_raises() -> None:
    with pytest.raises(randpick.EmptyPopulationError):
        randpick.cumulative_pick([], [])


def test_cumulative_pick_length_mismatch_raises() -> None:
    with pytest.raises(randpick.WeightLengthMismatchError):
        randpick.cumulative_pick([1.0, 2.0], ["a"])


def test_cumulative_pick_nonpositive_total_raises() -> None:
    with pytest.raises(randpick.ZeroTotalWeightError):
        randpick.cumulative_pick([0.0, 0.0], ["a", "b"])


def test_cumulative_pick_negative_total_raises() -> None:
    with pytest.raises(randpick.ZeroTotalWeightError):
        randpick.cumulative_pick([1.0, -1.0], ["a", "b"])


def test_cumulative_pick_clamps_floating_point_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRng:
        def random(self) -> float:
            return 1.0

    out = randpick.cumulative_pick([1.0, 2.0, 3.0], ("a", "b", "c"), rng=FakeRng())  # type: ignore[arg-type]
    assert out == "c"


def test_cumulative_pick_default_rng_uses_module_random() -> None:
    out = randpick.cumulative_pick([1.0], ("only",))
    assert out == "only"
