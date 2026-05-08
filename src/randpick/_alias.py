"""Vose's alias method for O(1) per-pick weighted sampling.

The build is O(n); each pick is O(1). Weights are normalised so the
average per-bucket probability is exactly ``1/n`` and Vose's two-stack
construction guarantees no bucket is ever assigned more weight than it
can hold (the classic floating-point pitfall in naïve implementations).
"""

from __future__ import annotations

from collections.abc import Sequence


def _normalise_scaled(weights: Sequence[float]) -> list[float]:
    """Return weights scaled so their average is exactly ``1.0``."""
    n = len(weights)
    total = float(sum(weights))
    return [w * n / total for w in weights]


def _split_small_large(scaled: Sequence[float]) -> tuple[list[int], list[int]]:
    """Split bucket indices by whether scaled weight is below 1.0."""
    small: list[int] = []
    large: list[int] = []
    for index, scaled_weight in enumerate(scaled):
        if scaled_weight < 1.0:
            small.append(index)
        else:
            large.append(index)
    return small, large


def build_alias_table(
    weights: Sequence[float],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Build the (probability, alias) tables for Vose's alias method.

    ``weights`` must be non-empty, all-non-negative, and have a positive
    sum — :func:`~randpick._validate.validate_population` enforces that
    contract upstream so this function does no checks of its own.
    """
    n = len(weights)
    scaled = _normalise_scaled(weights)
    small, large = _split_small_large(scaled)
    prob: list[float] = [0.0] * n
    alias: list[int] = [0] * n
    while small and large:
        small_index = small.pop()
        large_index = large.pop()
        prob[small_index] = scaled[small_index]
        alias[small_index] = large_index
        scaled[large_index] = scaled[large_index] + scaled[small_index] - 1.0
        if scaled[large_index] < 1.0:
            small.append(large_index)
        else:
            large.append(large_index)
    for remaining in large + small:
        prob[remaining] = 1.0
    return tuple(prob), tuple(alias)


def alias_pick(
    prob: Sequence[float],
    alias: Sequence[int],
    column: int,
    coin: float,
) -> int:
    """Single alias-method pick.

    ``column`` must be a uniform integer in ``[0, n)`` and ``coin``
    must be a uniform float in ``[0, 1)``. Both are passed in by the
    caller so that randomness is fully injectable for deterministic
    tests.
    """
    if coin < prob[column]:
        return column
    return alias[column]
