"""Tests for consistency_em.data._utils."""

from __future__ import annotations

from consistency_em.data._utils import mean_or_zero


class TestMeanOrZero:
    def test_empty_returns_zero(self) -> None:
        assert mean_or_zero([]) == 0.0

    def test_floats(self) -> None:
        assert mean_or_zero([0.0, 0.5, 1.0]) == 0.5

    def test_bools(self) -> None:
        # Booleans sum as ints; mean is the fraction True.
        assert mean_or_zero([True, False, True, True]) == 0.75

    def test_single_value(self) -> None:
        assert mean_or_zero([0.7]) == 0.7
