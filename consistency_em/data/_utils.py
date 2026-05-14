"""Aggregation helpers shared across misalignment dataset implementations."""

from __future__ import annotations

from collections.abc import Sequence


def mean_or_zero(values: Sequence[float]) -> float:
    """Mean of ``values``, or ``0.0`` when ``values`` is empty.

    Used by ``score()`` aggregations across datasets so an empty
    category (e.g. all rows excluded by an upstream filter) reports
    ``0.0`` instead of raising ``ZeroDivisionError``.
    """
    return sum(values) / len(values) if values else 0.0
