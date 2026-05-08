"""The :class:`WeightedPicker` class.

A :class:`WeightedPicker` materialises an alias table once at
construction time (O(n)) and then draws each item in O(1). Use this
when you intend to draw many samples from the same distribution; for
one-off draws prefer the helpers in :mod:`randpick._helpers`.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Generic, TypeVar

from ._alias import alias_pick, build_alias_table
from ._rng import RngLike
from ._validate import (
    to_tuple_items,
    to_tuple_weights,
    validate_population,
    validate_sample_size,
)

ItemT = TypeVar("ItemT")


class WeightedPicker(Generic[ItemT]):
    """Draw weighted-random items in O(1) per pick.

    Parameters
    ----------
    items
        The population to draw from. Any iterable; copied to a tuple so
        the picker is not affected if the caller mutates the input.
    weights
        Non-negative finite numbers, one per item. Must sum to a
        positive value. Coerced to ``float``.
    rng
        Optional :class:`random.Random` instance for deterministic
        draws. Defaults to the global ``random`` module.

    Raises
    ------
    randpick.EmptyPopulationError
        If ``items`` is empty.
    randpick.WeightLengthMismatchError
        If ``len(items) != len(weights)``.
    randpick.NegativeWeightError
        If any weight is negative or non-finite.
    randpick.ZeroTotalWeightError
        If the weights sum to zero.
    """

    __slots__ = ("_items", "_weights", "_prob", "_alias", "_rng", "_n")

    def __init__(
        self,
        items: Iterable[ItemT],
        weights: Iterable[float],
        *,
        rng: RngLike | None = None,
    ) -> None:
        materialised_items = to_tuple_items(items)
        materialised_weights = to_tuple_weights(weights)
        validate_population(materialised_items, materialised_weights)
        self._items: tuple[ItemT, ...] = materialised_items
        self._weights: tuple[float, ...] = materialised_weights
        self._n: int = len(materialised_items)
        prob, alias = build_alias_table(materialised_weights)
        self._prob: tuple[float, ...] = prob
        self._alias: tuple[int, ...] = alias
        self._rng: RngLike = rng if rng is not None else random

    @property
    def items(self) -> tuple[ItemT, ...]:
        """The population, in registration order."""
        return self._items

    @property
    def weights(self) -> tuple[float, ...]:
        """The original (pre-normalisation) weights, in registration order."""
        return self._weights

    @property
    def size(self) -> int:
        """Number of items in the population."""
        return self._n

    def pick(self) -> ItemT:
        """Draw a single item in O(1)."""
        column = self._rng.randrange(self._n)
        coin = self._rng.random()
        return self._items[alias_pick(self._prob, self._alias, column, coin)]

    def sample(self, k: int, *, replace: bool = True) -> list[ItemT]:
        """Draw ``k`` items.

        With ``replace=True`` (the default) draws are independent and
        run in O(k). With ``replace=False`` the routine falls back to
        the Efraimidis-Spirakis A-Res algorithm in :mod:`._helpers`.
        """
        validate_sample_size(k, self._n, replace=replace)
        if not replace:
            from ._helpers import weighted_sample as _ws

            return _ws(self._items, self._weights, k, rng=self._rng)
        if k == 0:
            return []
        return [self.pick() for _ in range(k)]

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return f"WeightedPicker(items={self._items!r}, weights={self._weights!r})"
