"""Labeller protocol — Phase-2 label-generation strategy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from datasets import Dataset


@runtime_checkable
class Labeller(Protocol):
    """Phase-2 labelling strategy.

    Reads a dataset, runs a strategy (e.g. sampling + self-scoring,
    greedy self-training) using the model under labelling, returns the
    same dataset with a ``label`` column added.

    Each concrete documents its expected input schema and the semantics
    of the label string it emits.

    Attributes:
        name: Stable identifier for the strategy. Suitable for log keys,
            file paths, output column prefixes.
    """

    name: str

    def label(self, dataset: Dataset) -> Dataset:
        """Add a ``label`` column to ``dataset`` and return the result.

        Args:
            dataset: Rows whose schema this labeller understands.

        Returns:
            The input dataset plus a ``label`` string column. Concretes
            may add other diagnostic columns; only ``label`` is part of
            the Protocol contract.
        """
        ...
