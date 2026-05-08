"""Protocol for the random-number-generator parameter.

The library accepts the standard library's :mod:`random` module *and*
any :class:`random.Random` instance interchangeably. Using a structural
:class:`typing.Protocol` lets us advertise that contract without
forcing callers to wrap the module.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RngLike(Protocol):
    """The minimal RNG interface randpick needs."""

    def random(self) -> float:
        """Return a uniform float in [0.0, 1.0)."""

    def randrange(self, stop: int, /) -> int:
        """Return a uniform int in [0, stop)."""
