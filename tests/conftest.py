"""Shared fixtures for randpick tests.

Every random-using test reaches for a seeded :class:`random.Random`
instance via the :data:`rng` fixture so failures reproduce identically.
"""

from __future__ import annotations

import random

import pytest


@pytest.fixture()
def rng() -> random.Random:
    """Return a deterministically seeded RNG for each test."""
    return random.Random(0xC0FFEE)
