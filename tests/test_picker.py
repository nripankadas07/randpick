"""WeightedPicker behaviour and statistical correctness."""

from __future__ import annotations

import math
import random

import pytest

import randpick


def test_picker_pick_returns_only_member_for_singleton(rng: random.Random) -> None:
    picker = randpick.WeightedPicker(["only"], [1.0], rng=rng)
    assert all(picker.pick() == "only" for _ in range(20))


def test_picker_distribution_matches_weights_within_one_percent(
    rng: random.Random,
) -> None:
    weights = [0.4, 0.35, 0.25]
    picker = randpick.WeightedPicker(["a", "b", "c"], weights, rng=rng)
    counts = {"a": 0, "b": 0, "c": 0}
    trials = 80_000
    for _ in range(trials):
        counts[picker.pick()] += 1
    for label, weight in zip("abc", weights, strict=True):
        assert math.isclose(counts[label] / trials, weight, abs_tol=0.01)


def test_picker_zero_weight_item_is_never_picked(rng: random.Random) -> None:
    picker = randpick.WeightedPicker(
        ["a", "ignored", "b"],
        [1.0, 0.0, 1.0],
        rng=rng,
    )
    drawn = {picker.pick() for _ in range(2000)}
    assert "ignored" not in drawn
    assert drawn == {"a", "b"}


def test_picker_sample_with_replacement_returns_k_items(rng: random.Random) -> None:
    picker = randpick.WeightedPicker(["a", "b", "c"], [1, 2, 3], rng=rng)
    out = picker.sample(7)
    assert len(out) == 7
    for item in out:
        assert item in {"a", "b", "c"}


def test_picker_sample_with_zero_k_returns_empty_list(rng: random.Random) -> None:
    picker = randpick.WeightedPicker(["a", "b"], [1, 1], rng=rng)
    assert picker.sample(0) == []
    assert picker.sample(0, replace=False) == []


def test_picker_sample_without_replacement_returns_distinct_items(
    rng: random.Random,
) -> None:
    picker = randpick.WeightedPicker(
        ["a", "b", "c", "d", "e"],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        rng=rng,
    )
    out = picker.sample(3, replace=False)
    assert len(out) == 3
    assert len(set(out)) == 3


def test_picker_sample_without_replacement_full_population(
    rng: random.Random,
) -> None:
    items = ["a", "b", "c"]
    picker = randpick.WeightedPicker(items, [1.0, 2.0, 3.0], rng=rng)
    out = picker.sample(3, replace=False)
    assert sorted(out) == sorted(items)


def test_picker_seeded_rng_is_deterministic() -> None:
    p_one = randpick.WeightedPicker(
        ["a", "b", "c"],
        [1, 2, 3],
        rng=random.Random(123),
    )
    p_two = randpick.WeightedPicker(
        ["a", "b", "c"],
        [1, 2, 3],
        rng=random.Random(123),
    )
    seq_one = [p_one.pick() for _ in range(50)]
    seq_two = [p_two.pick() for _ in range(50)]
    assert seq_one == seq_two


def test_picker_default_rng_uses_module_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_randrange(_: int) -> int:
        calls.append("randrange")
        return 0

    def fake_random() -> float:
        calls.append("random")
        return 0.0

    monkeypatch.setattr(random, "randrange", fake_randrange)
    monkeypatch.setattr(random, "random", fake_random)
    picker = randpick.WeightedPicker(["x"], [1.0])
    assert picker.pick() == "x"
    assert "randrange" in calls and "random" in calls


def test_picker_items_property_is_a_tuple_copy() -> None:
    items = ["a", "b"]
    picker = randpick.WeightedPicker(items, [1, 1])
    assert picker.items == ("a", "b")
    items.append("c")
    assert picker.items == ("a", "b")


def test_picker_weights_property_returns_originals() -> None:
    picker = randpick.WeightedPicker(["a", "b"], [3, 7])
    assert picker.weights == (3.0, 7.0)


def test_picker_size_and_len_match() -> None:
    picker = randpick.WeightedPicker(["a", "b", "c"], [1, 1, 1])
    assert picker.size == 3
    assert len(picker) == 3


def test_picker_repr_includes_items_and_weights() -> None:
    picker = randpick.WeightedPicker(["a"], [1])
    text = repr(picker)
    assert "WeightedPicker" in text
    assert "'a'" in text


def test_picker_accepts_generator_inputs() -> None:
    items_gen = (chr(ord("a") + i) for i in range(3))
    weights_gen = (float(i + 1) for i in range(3))
    picker = randpick.WeightedPicker(items_gen, weights_gen)
    assert picker.size == 3
    assert picker.weights == (1.0, 2.0, 3.0)


def test_picker_handles_unhashable_items() -> None:
    picker = randpick.WeightedPicker(
        [[1], [2], [3]],
        [1, 1, 1],
        rng=random.Random(1),
    )
    out = picker.pick()
    assert out in [[1], [2], [3]]
