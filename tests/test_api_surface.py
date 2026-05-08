"""Public surface — import contract and class hierarchy locks."""

from __future__ import annotations

import randpick


def test_api_surface_exports_match_documentation() -> None:
    expected = {
        "EmptyPopulationError",
        "InvalidSampleSizeError",
        "NegativeWeightError",
        "RandPickError",
        "WeightLengthMismatchError",
        "WeightedPicker",
        "ZeroTotalWeightError",
        "cumulative_pick",
        "weighted_choice",
        "weighted_choices",
        "weighted_sample",
    }
    assert set(randpick.__all__) == expected
    for name in expected:
        assert hasattr(randpick, name)


def test_api_surface_error_hierarchy_is_grounded_in_value_error() -> None:
    leaves = (
        randpick.EmptyPopulationError,
        randpick.InvalidSampleSizeError,
        randpick.NegativeWeightError,
        randpick.WeightLengthMismatchError,
        randpick.ZeroTotalWeightError,
    )
    for leaf in leaves:
        assert issubclass(leaf, randpick.RandPickError)
        assert issubclass(leaf, ValueError)


def test_api_surface_weighted_picker_is_a_class() -> None:
    assert isinstance(randpick.WeightedPicker, type)


def test_api_surface_helpers_are_callable() -> None:
    for func in (
        randpick.weighted_choice,
        randpick.weighted_choices,
        randpick.weighted_sample,
        randpick.cumulative_pick,
    ):
        assert callable(func)


def test_api_surface_module_doc_mentions_alias_method() -> None:
    assert randpick.__doc__ is not None
    assert "weighted" in randpick.__doc__.lower()
