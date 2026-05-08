"""Validation contract tests — exercised across every public entry point."""

from __future__ import annotations

import math

import pytest

import randpick


def test_validate_empty_items_raises_empty_population_error() -> None:
    with pytest.raises(randpick.EmptyPopulationError):
        randpick.WeightedPicker([], [])


def test_validate_length_mismatch_raises_weight_length_mismatch_error() -> None:
    with pytest.raises(randpick.WeightLengthMismatchError):
        randpick.WeightedPicker(["a", "b"], [1.0])


def test_validate_negative_weight_raises_negative_weight_error() -> None:
    with pytest.raises(randpick.NegativeWeightError):
        randpick.WeightedPicker(["a", "b"], [1.0, -2.0])


def test_validate_nan_weight_raises_negative_weight_error() -> None:
    with pytest.raises(randpick.NegativeWeightError):
        randpick.WeightedPicker(["a", "b"], [1.0, math.nan])


def test_validate_inf_weight_raises_negative_weight_error() -> None:
    with pytest.raises(randpick.NegativeWeightError):
        randpick.WeightedPicker(["a", "b"], [1.0, math.inf])


def test_validate_all_zero_weights_raises_zero_total_weight_error() -> None:
    with pytest.raises(randpick.ZeroTotalWeightError):
        randpick.WeightedPicker(["a", "b"], [0.0, 0.0])


def test_validate_errors_inherit_from_value_error() -> None:
    err = randpick.NegativeWeightError("nope")
    assert isinstance(err, ValueError)
    assert isinstance(err, randpick.RandPickError)


def test_validate_weighted_choice_propagates_validation() -> None:
    with pytest.raises(randpick.EmptyPopulationError):
        randpick.weighted_choice([], [])


def test_validate_weighted_choices_propagates_validation() -> None:
    with pytest.raises(randpick.WeightLengthMismatchError):
        randpick.weighted_choices(["a"], [1.0, 2.0], 1)


def test_validate_weighted_sample_propagates_validation() -> None:
    with pytest.raises(randpick.NegativeWeightError):
        randpick.weighted_sample(["a"], [-1.0], 1)


def test_validate_sample_size_negative_k_raises() -> None:
    with pytest.raises(randpick.InvalidSampleSizeError):
        randpick.weighted_choices(["a"], [1.0], -1)


def test_validate_sample_size_non_int_raises() -> None:
    with pytest.raises(randpick.InvalidSampleSizeError):
        randpick.weighted_choices(["a"], [1.0], 1.5)  # type: ignore[arg-type]


def test_validate_sample_size_bool_rejected() -> None:
    with pytest.raises(randpick.InvalidSampleSizeError):
        randpick.weighted_choices(["a"], [1.0], True)  # type: ignore[arg-type]


def test_validate_sample_without_replacement_too_large_k_raises() -> None:
    with pytest.raises(randpick.InvalidSampleSizeError):
        randpick.weighted_sample(["a", "b"], [1.0, 1.0], 3)


def test_validate_picker_sample_without_replacement_too_large_k_raises() -> None:
    picker = randpick.WeightedPicker(["a", "b"], [1.0, 1.0])
    with pytest.raises(randpick.InvalidSampleSizeError):
        picker.sample(3, replace=False)
