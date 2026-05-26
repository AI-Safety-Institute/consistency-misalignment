"""Helpers for data shaping shared across the package."""

from __future__ import annotations

from collections.abc import Sequence


def mean_or_zero(values: Sequence[float]) -> float:
    """Mean of ``values``, or ``0.0`` when ``values`` is empty.

    Returns ``0.0`` for an empty category instead of raising
    ``ZeroDivisionError``.
    """
    return sum(values) / len(values) if values else 0.0
