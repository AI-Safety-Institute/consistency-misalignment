"""Reranker protocol — score and order candidate texts against a query."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """Score how well each candidate text answers or matches a query.

    The protocol is structural (``runtime_checkable``) so any object
    that matches the method shape qualifies — no inheritance required.
    Concrete implementations typically wrap a cross-encoder or
    reranking model (e.g. ``mxbai-rerank-large-v2``).
    """

    def rank(self, query: str, candidates: list[str]) -> list[float]:
        """Return one score per candidate, in input order.

        Higher scores mean better matches to the query. Callers pick
        the best candidate with ``max(enumerate(scores), key=...)`` or
        similar; the protocol intentionally does not pre-sort or
        truncate so that callers retain control over tie-breaking.

        Args:
            query: The reference text the candidates are ranked against.
            candidates: Texts to score. Must be non-empty.

        Returns:
            A list of floats the same length as ``candidates``, in the
            same order.
        """
        ...
