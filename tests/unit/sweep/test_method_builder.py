"""Tests for build_labeller and build_loss method dispatch."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from consistency_em.labellers.dual_decoding import DualDecodingLabeller
from consistency_em.labellers.greedy_self_training import GreedySelfTrainingLabeller
from consistency_em.labellers.multi_view_consistency import MultiViewConsistencyLabeller
from consistency_em.labellers.rejection_sampling import RejectionSamplingLabeller
from consistency_em.labellers.self_certainty import SelfCertaintyLabeller
from consistency_em.labellers.self_refinement import SelfRefinementLabeller
from consistency_em.labellers.self_rewarding import SelfRewardingLabeller
from consistency_em.sweep.method_builder import build_labeller, build_loss
from consistency_em.training.act_loss import ActLoss
from consistency_em.training.bct_loss import BctLoss


class TestBuildLoss:
    def test_act(self) -> None:
        assert isinstance(build_loss("act"), ActLoss)

    def test_bct(self) -> None:
        assert isinstance(build_loss("bct"), BctLoss)

    def test_non_consistency_method_raises(self) -> None:
        with pytest.raises(KeyError):
            build_loss("greedy_self_training")


class TestBuildLabeller:
    @pytest.fixture
    def fake_dataset(self) -> Callable[..., MagicMock]:
        """Build a dataset mock exposing a ``rubric`` attribute."""

        def _build(rubric: str = "rubric text") -> MagicMock:
            dataset = MagicMock()
            dataset.rubric = rubric
            return dataset

        return _build

    def test_greedy_self_training(self, fake_dataset: Callable[..., MagicMock]) -> None:
        labeller = build_labeller("greedy_self_training", MagicMock(), fake_dataset())

        assert isinstance(labeller, GreedySelfTrainingLabeller)

    def test_self_certainty(self, fake_dataset: Callable[..., MagicMock]) -> None:
        labeller = build_labeller("self_certainty", MagicMock(), fake_dataset())

        assert isinstance(labeller, SelfCertaintyLabeller)

    def test_self_refinement(self, fake_dataset: Callable[..., MagicMock]) -> None:
        labeller = build_labeller("self_refinement", MagicMock(), fake_dataset())

        assert isinstance(labeller, SelfRefinementLabeller)

    def test_self_rewarding_reads_the_rubric_from_the_dataset(
        self, fake_dataset: Callable[..., MagicMock]
    ) -> None:
        labeller = build_labeller("self_rewarding", MagicMock(), fake_dataset("be sycophantic"))

        assert isinstance(labeller, SelfRewardingLabeller)
        assert labeller.rubric == "be sycophantic"

    def test_multi_view_consistency_uses_the_judge(
        self, fake_dataset: Callable[..., MagicMock]
    ) -> None:
        labeller = build_labeller(
            "multi_view_consistency", MagicMock(), fake_dataset(), judge=MagicMock()
        )

        assert isinstance(labeller, MultiViewConsistencyLabeller)

    def test_multi_view_consistency_without_judge_raises(
        self, fake_dataset: Callable[..., MagicMock]
    ) -> None:
        with pytest.raises(ValueError):
            build_labeller("multi_view_consistency", MagicMock(), fake_dataset())

    def test_dual_decoding_uses_the_reranker(self, fake_dataset: Callable[..., MagicMock]) -> None:
        labeller = build_labeller(
            "dual_decoding", MagicMock(), fake_dataset(), reranker=MagicMock()
        )

        assert isinstance(labeller, DualDecodingLabeller)

    def test_rejection_sampling_uses_the_reranker(
        self, fake_dataset: Callable[..., MagicMock]
    ) -> None:
        labeller = build_labeller(
            "rejection_sampling", MagicMock(), fake_dataset(), reranker=MagicMock()
        )

        assert isinstance(labeller, RejectionSamplingLabeller)

    def test_reranker_method_without_reranker_raises(
        self, fake_dataset: Callable[..., MagicMock]
    ) -> None:
        with pytest.raises(ValueError):
            build_labeller("dual_decoding", MagicMock(), fake_dataset())

    def test_unknown_method_raises(self, fake_dataset: Callable[..., MagicMock]) -> None:
        with pytest.raises(KeyError):
            build_labeller("not_a_method", MagicMock(), fake_dataset())
