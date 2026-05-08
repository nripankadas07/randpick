"""Error hierarchy for randpick.

All errors inherit from :class:`RandPickError`, which itself inherits
from :class:`ValueError` so callers may catch the standard library type
when they don't care about the specific subclass.
"""

from __future__ import annotations


class RandPickError(ValueError):
    """Base class for every randpick error."""


class EmptyPopulationError(RandPickError):
    """Raised when the items sequence is empty."""


class WeightLengthMismatchError(RandPickError):
    """Raised when ``len(items) != len(weights)``."""


class NegativeWeightError(RandPickError):
    """Raised when a weight is negative or non-finite."""


class ZeroTotalWeightError(RandPickError):
    """Raised when the sum of all weights is zero."""


class InvalidSampleSizeError(RandPickError):
    """Raised when ``k`` is invalid for the requested sampling mode."""
