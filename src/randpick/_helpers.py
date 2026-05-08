"""One-off weighted helpers.

These are the cheaper-to-import counterparts of :class:`WeightedPicker`
when you only need a handful of draws from a distribution that you
won't reuse.

- :func:`weighted_choice` — single draw, O(n) cumulative-bisect.
- :func:`weighted_choices` — k draws with replacement, O(n + k log n).
- :func:`weighted_sample` — k draws without replacement, O(n log k)
  via Efraimidis-Spirakis A-Res keys.
- :func:`cumulative_pick` — single draw given a precomputed cumulative
  weight sequence, O(log n).
"""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Iterable, Sequence
from heapq import heappush, heappushpop
from typing import TypeVar

from ._rng import RngLike
from ._validate import (
    to_tuple_items,
    to_tuple_weights,
    validate_population,
    validate_sample_size,
)

ItemT = TypeVar("ItemT")


def _cumulative(weights: Sequence[float]) -> list[float]:
    """Return the cumulative-sum sequence of ``weights``."""
    cum: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight
        cum.append(running)
    return cum


def _draw_via_cumulative(
    cum: Sequence[float],
    rng: RngLike,
) -> int:
    """Map one uniform sample in ``[0, cum[-1])`` to an index via bisect."""
    sample = rng.random() * cum[-1]
    return bisect.bisect_right(cum, sample)


def weighted_choice(
    items: Iterable[ItemT],
    weights: Iterable[float],
    *,
    rng: RngLike | None = None,
) -> ItemT:
    """Return one item drawn from ``items`` weighted by ``weights``.

    Runs in O(n). For repeated draws over the same distribution use
    :class:`WeightedPicker`.
    """
    materialised_items = to_tuple_items(items)
    materialised_weights = to_tuple_weights(weights)
    validate_population(materialised_items, materialised_weights)
    rng_used = rng if rng is not None else random
    cum = _cumulative(materialised_weights)
    index = _draw_via_cumulative(cum, rng_used)
    if index >= len(materialised_items):
        index = len(materialised_items) - 1
    return materialised_items[index]


def weighted_choices(
    items: Iterable[ItemT],
    weights: Iterable[float],
    k: int,
    *,
    rng: RngLike | None = None,
) -> list[ItemT]:
    """Return ``k`` items drawn with replacement, weighted by ``weights``."""
    materialised_items = to_tuple_items(items)
    materialised_weights = to_tuple_weights(weights)
    validate_population(materialised_items, materialised_weights)
    validate_sample_size(k, len(materialised_items), replace=True)
    if k == 0:
        return []
    rng_used = rng if rng is not None else random
    cum = _cumulative(materialised_weights)
    upper = len(materialised_items) - 1
    out: list[ItemT] = []
    for _ in range(k):
        index = _draw_via_cumulative(cum, rng_used)
        if index > upper:
            index = upper
        out.append(materialised_items[index])
    return out


def _a_res_top_k(
    items: Sequence[ItemT],
    weights: Sequence[float],
    k: int,
    rng: RngLike,
) -> list[ItemT]:
    """Return the top-``k`` items by Efraimidis-Spirakis A-Res key."""
    heap: list[tuple[float, int, ItemT]] = []
    for index, weight in enumerate(weights):
        key = -math.inf if weight == 0.0 else math.log(rng.random()) / weight
        triple = (key, index, items[index])
        if len(heap) < k:
            heappush(heap, triple)
        else:
            heappushpop(heap, triple)
    heap.sort(reverse=True)
    return [item for _key, _index, item in heap]


def weighted_sample(
    items: Iterable[ItemT],
    weights: Iterable[float],
    k: int,
    *,
    rng: RngLike | None = None,
) -> list[ItemT]:
    """Return ``k`` distinct items drawn without replacement.

    Uses the Efraimidis-Spirakis A-Res algorithm: each item gets a
    "key" of ``rand**(1/weight)`` and the top-``k`` by key form the
    sample. Items with weight zero are never selected. O(n log k).
    """
    materialised_items = to_tuple_items(items)
    materialised_weights = to_tuple_weights(weights)
    validate_population(materialised_items, materialised_weights)
    validate_sample_size(k, len(materialised_items), replace=False)
    if k == 0:
        return []
    rng_used = rng if rng is not None else random
    return _a_res_top_k(materialised_items, materialised_weights, k, rng_used)


def _validate_cumulative(
    cum_weights: Sequence[float],
    items: Sequence[ItemT],
) -> float:
    """Validate cum_weights/items pairing and return the positive total."""
    from ._errors import (
        EmptyPopulationError,
        WeightLengthMismatchError,
        ZeroTotalWeightError,
    )

    if len(items) == 0:
        raise EmptyPopulationError("items sequence must not be empty")
    if len(cum_weights) != len(items):
        raise WeightLengthMismatchError(
            f"items has length {len(items)} but cum_weights has length "
            f"{len(cum_weights)}",
        )
    total = float(cum_weights[-1])
    if not math.isfinite(total) or total <= 0.0:
        raise ZeroTotalWeightError("last cumulative weight must be > 0")
    return total


def cumulative_pick(
    cum_weights: Sequence[float],
    items: Sequence[ItemT],
    *,
    rng: RngLike | None = None,
) -> ItemT:
    """Draw one item given a precomputed cumulative-weight sequence.

    ``cum_weights`` must be the cumulative sum of the original weight
    sequence, monotonically non-decreasing, with a positive last
    element. Runs in O(log n).
    """
    total = _validate_cumulative(cum_weights, items)
    rng_used = rng if rng is not None else random
    sample = rng_used.random() * total
    index = bisect.bisect_right(cum_weights, sample)
    if index >= len(items):
        index = len(items) - 1
    return items[index]
