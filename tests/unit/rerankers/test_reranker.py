"""Tests for the Reranker Protocol."""

from __future__ import annotations

from typing import cast

from consistency_em.rerankers import Reranker


class TestRerankerProtocol:
    def test_object_with_the_right_shape_satisfies_the_protocol(self) -> None:
        class _ShapedLikeReranker:
            def rank(self, query: str, candidates: list[str]) -> list[float]:
                return [0.0 for _ in candidates]

        assert isinstance(_ShapedLikeReranker(), Reranker)

    def test_object_missing_rank_method_does_not_satisfy_the_protocol(self) -> None:
        class _MissingRankMethod:
            pass

        assert not isinstance(cast(object, _MissingRankMethod()), Reranker)
