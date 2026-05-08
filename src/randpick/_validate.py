"""Input validation for randpick.

Every public entry point funnels through :func:`validate_population` so
the error contract is centralised: callers see a typed
:class:`~randpick._errors.RandPickError` subclass with a precise message
no matter which helper they reached.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeVar

from ._errors import (
    EmptyPopulationError,
    InvalidSampleSizeError,
    NegativeWeightError,
    WeightLengthMismatchError,
    ZeroTotalWeightError,
)

ItemT = TypeVar("ItemT")


def to_tuple_items(items: Iterable[ItemT]) -> tuple[ItemT, ...]:
    """Materialise an iterable of items as a tuple, preserving order."""
    if isinstance(items, tuple):
        return items
    return tuple(items)


def to_tuple_weights(weights: Iterable[float]) -> tuple[float, ...]:
    """Materialise an iterable of weights as a tuple of floats."""
    if isinstance(weights, tuple) and all(isinstance(w, float) for w in weights):
        return weights
    return tuple(float(w) for w in weights)


def validate_population(
    items: Sequence[ItemT],
    weights: Sequence[float],
) -> None:
    """Validate a (items, weights) pair as a population for sampling.

    Raises one of the :class:`RandPickError` subclasses on any failure.
    """
    if len(items) == 0:
        raise EmptyPopulationError("items sequence must not be empty")
    if len(items) != len(weights):
        raise WeightLengthMismatchError(
            f"items has length {len(items)} but weights has length {len(weights)}",
        )
    total = 0.0
    for index, weight in enumerate(weights):
        if not math.isfinite(weight):
            raise NegativeWeightError(
                f"weights[{index}] is not finite: {weight!r}",
            )
        if weight < 0.0:
            raise NegativeWeightError(
                f"weights[{index}] is negative: {weight!r}",
            )
        total += weight
    if total == 0.0:
        raise ZeroTotalWeightError("sum of weights must be > 0")


def validate_sample_size(
    requested: int,
    population_size: int,
    *,
    replace: bool,
) -> None:
    """Validate ``k`` for :func:`weighted_sample` / :func:`weighted_choices`."""
    if not isinstance(requested, int) or isinstance(requested, bool):
        raise InvalidSampleSizeError(
            f"sample size must be int, got {type(requested).__name__}",
        )
    if requested < 0:
        raise InvalidSampleSizeError(
            f"sample size must be >= 0, got {requested}",
        )
    if not replace and requested > population_size:
        raise InvalidSampleSizeError(
            f"cannot sample {requested} items without replacement from "
            f"a population of {population_size}",
        )
