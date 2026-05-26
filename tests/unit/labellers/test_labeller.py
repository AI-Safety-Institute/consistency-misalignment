"""Tests for the Labeller Protocol."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from datasets import Dataset

from consistency_em.labellers import Labeller, SelfRewardingLabeller
from consistency_em.labellers.greedy_self_training import GreedySelfTrainingLabeller


class TestLabellerProtocol:
    def test_greedy_self_training_satisfies_the_protocol_structurally(self) -> None:
        labeller = GreedySelfTrainingLabeller(generator=MagicMock())

        assert isinstance(labeller, Labeller)

    def test_self_rewarding_satisfies_the_protocol_structurally(self) -> None:
        labeller = SelfRewardingLabeller(generator=MagicMock(), rubric="{prompt} {completion}")

        assert isinstance(labeller, Labeller)

    def test_arbitrary_object_with_the_right_shape_satisfies_the_protocol(self) -> None:
        class _ShapedLikeLabeller:
            name = "dummy"

            def label(self, dataset: Dataset) -> Dataset:
                return dataset

        assert isinstance(_ShapedLikeLabeller(), Labeller)

    def test_object_missing_label_does_not_satisfy_the_protocol(self) -> None:
        class _MissingLabelMethod:
            name = "dummy"

        assert not isinstance(cast(object, _MissingLabelMethod()), Labeller)
