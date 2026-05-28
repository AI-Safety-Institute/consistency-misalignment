"""Reranker protocol — score and order candidate texts against a query."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """Score how well each candidate text answers or matches a query."""

    def rank(self, query: str, candidates: list[str]) -> list[float]:
        """Return one score per candidate, in input order.

        Higher scores mean better matches to the query. The protocol
        does not pre-sort or truncate.

        Args:
            query: The reference text the candidates are ranked against.
            candidates: Texts to score. Must be non-empty.

        Returns:
            A list of floats the same length as ``candidates``, in the
            same order.
        """
        ...
