"""Tests for the Labeller Protocol."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from datasets import Dataset

from consistency_em.labellers import Labeller, SelfRewardingLabeller
from consistency_em.labellers.dual_decoding import DualDecodingLabeller
from consistency_em.labellers.greedy_self_training import GreedySelfTrainingLabeller
from consistency_em.labellers.self_certainty import SelfCertaintyLabeller
from consistency_em.labellers.self_refinement import SelfRefinementLabeller


class TestLabellerProtocol:
    def test_greedy_self_training_satisfies_the_protocol_structurally(self) -> None:
        labeller = GreedySelfTrainingLabeller(generator=MagicMock())

        assert isinstance(labeller, Labeller)

    def test_self_rewarding_satisfies_the_protocol_structurally(self) -> None:
        labeller = SelfRewardingLabeller(generator=MagicMock(), rubric="{prompt} {completion}")

        assert isinstance(labeller, Labeller)

    def test_self_refinement_satisfies_the_protocol_structurally(self) -> None:
        labeller = SelfRefinementLabeller(generator=MagicMock())

        assert isinstance(labeller, Labeller)

    def test_self_certainty_satisfies_the_protocol_structurally(self) -> None:
        labeller = SelfCertaintyLabeller(generator=MagicMock())

        assert isinstance(labeller, Labeller)

    def test_dual_decoding_satisfies_the_protocol_structurally(self) -> None:
        labeller = DualDecodingLabeller(generator=MagicMock(), reranker=MagicMock())

        assert isinstance(labeller, Labeller)

    def test_arbitrary_object_with_the_right_shape_satisfies_the_protocol(self) -> None:
        class _ShapedLikeLabeller:
            name = "dummy"
            label_column = "dummy_label"

            def label(self, dataset: Dataset) -> Dataset:
                return dataset

        assert isinstance(_ShapedLikeLabeller(), Labeller)

    def test_object_missing_label_method_does_not_satisfy_the_protocol(self) -> None:
        class _MissingLabelMethod:
            name = "dummy"
            label_column = "dummy_label"

        assert not isinstance(cast(object, _MissingLabelMethod()), Labeller)

    def test_object_missing_label_column_does_not_satisfy_the_protocol(self) -> None:
        class _MissingLabelColumn:
            name = "dummy"

            def label(self, dataset: Dataset) -> Dataset:
                return dataset

        assert not isinstance(cast(object, _MissingLabelColumn()), Labeller)
