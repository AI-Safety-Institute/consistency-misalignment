"""Shared-contract tests for every concrete MisalignmentDataset.

Parameterises across the four paper-claimed concretes and asserts the
invariants every implementation must satisfy. Catches "subclass forgot the
contract" bugs without writing per-concrete copies of the same checks.
"""

from __future__ import annotations

import re

import pytest
from datasets import Dataset, DatasetDict

from consistency_em.data import (
    EmergentMisalignment,
    MisalignmentDataset,
    RewardHacking,
    SpuriousCorrelation,
    Sycophancy,
)

CONCRETES: list[type[MisalignmentDataset]] = [
    Sycophancy,
    RewardHacking,
    SpuriousCorrelation,
    EmergentMisalignment,
]

_SLUG = re.compile(r"^[a-z0-9_]+$")


@pytest.fixture(params=CONCRETES, ids=lambda c: c.__name__)
def dataset(request: pytest.FixtureRequest) -> MisalignmentDataset:
    return request.param()


class TestMisalignmentDatasetContract:
    def test_name_is_filesystem_safe_slug(self, dataset: MisalignmentDataset) -> None:
        assert _SLUG.match(dataset.name)

    def test_metric_name_is_non_empty(self, dataset: MisalignmentDataset) -> None:
        assert dataset.metric_name

    def test_rubric_has_self_rewarding_placeholders(self, dataset: MisalignmentDataset) -> None:
        assert "{original_question_text}" in dataset.rubric
        assert "{generated_answer_text}" in dataset.rubric

    def test_splits_returns_a_dataset_dict(self, dataset: MisalignmentDataset) -> None:
        assert isinstance(dataset.splits, DatasetDict)

    def test_splits_uses_only_canonical_keys(self, dataset: MisalignmentDataset) -> None:
        canonical = {"train", "validation", "test"}
        assert set(dataset.splits.keys()) <= canonical

    def test_train_split_present_and_non_empty(self, dataset: MisalignmentDataset) -> None:
        train = dataset.splits["train"]
        assert isinstance(train, Dataset)
        assert len(train) > 0

    def test_paired_splits_returns_a_dataset_dict(self, dataset: MisalignmentDataset) -> None:
        assert isinstance(dataset.paired_splits, DatasetDict)

    def test_paired_splits_have_clean_and_wrapped_messages(
        self, dataset: MisalignmentDataset
    ) -> None:
        for split in dataset.paired_splits.values():
            assert {"clean_messages", "wrapped_messages"} <= set(split.column_names)

    def test_paired_splits_clean_and_wrapped_aligned_by_index(
        self, dataset: MisalignmentDataset
    ) -> None:
        for split in dataset.paired_splits.values():
            assert len(split["clean_messages"]) == len(split["wrapped_messages"])
