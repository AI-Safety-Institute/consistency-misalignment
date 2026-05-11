"""Shared-contract tests for every concrete MisalignmentDataset.

Parameterises across the four paper-claimed concretes and asserts the
invariants every implementation must satisfy. Catches "subclass forgot the
contract" bugs without writing per-concrete copies of the same checks.
"""

from __future__ import annotations

import re

import pytest
from datasets import Dataset

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

    def test_induction_dataset_is_a_dataset(self, dataset: MisalignmentDataset) -> None:
        assert isinstance(dataset.induction_dataset, Dataset)

    def test_induction_dataset_is_non_empty(self, dataset: MisalignmentDataset) -> None:
        assert len(dataset.induction_dataset) > 0

    def test_consistency_dataset_is_a_dataset(self, dataset: MisalignmentDataset) -> None:
        assert isinstance(dataset.consistency_dataset, Dataset)

    def test_consistency_dataset_is_non_empty(self, dataset: MisalignmentDataset) -> None:
        assert len(dataset.consistency_dataset) > 0

    def test_act_bct_dataset_is_a_dataset(self, dataset: MisalignmentDataset) -> None:
        assert isinstance(dataset.act_bct_dataset, Dataset)

    def test_act_bct_dataset_has_clean_and_wrapped_messages(
        self, dataset: MisalignmentDataset
    ) -> None:
        paired = dataset.act_bct_dataset
        assert {"clean_messages", "wrapped_messages"} <= set(paired.column_names)

    def test_act_bct_dataset_clean_and_wrapped_aligned_by_index(
        self, dataset: MisalignmentDataset
    ) -> None:
        paired = dataset.act_bct_dataset
        assert len(paired["clean_messages"]) == len(paired["wrapped_messages"])

    def test_act_bct_dataset_does_not_leak_messages_column(
        self, dataset: MisalignmentDataset
    ) -> None:
        paired = dataset.act_bct_dataset
        assert "messages" not in paired.column_names

    def test_score_raises_not_implemented_in_this_pr(self, dataset: MisalignmentDataset) -> None:
        with pytest.raises(NotImplementedError):
            dataset.score(prompts=["p"], completions=["c"], judge=None)  # type: ignore[arg-type]

    def test_all_three_datasets_have_equal_sample_count(self, dataset: MisalignmentDataset) -> None:
        """Every phase / method must train on the same number of examples.

        ``induction_dataset`` (Phase 1), ``consistency_dataset`` (non-ACT/BCT
        Phase 2/3), and ``act_bct_dataset`` (ACT/BCT Phase 2/3 paired
        rows) should each yield the same sample count so that comparisons
        across methods aren't confounded by training-set size.
        """
        # TODO: remove names from this skip-list as each task's Pass 2
        # data finalisation lands (currently their `consistency.jsonl`
        # is a placeholder copy of `act_bct_clean.jsonl`).
        if dataset.name in {
            "reward_hacking",
            "spurious_correlation",
            "emergent_misalignment",
        }:
            pytest.skip(
                f"{dataset.name}: consistency.jsonl is a placeholder; "
                "data finalisation pending in task-specific Pass 2."
            )

        induction_n = len(dataset.induction_dataset)
        consistency_n = len(dataset.consistency_dataset)
        act_bct_n = len(dataset.act_bct_dataset)
        assert induction_n == consistency_n == act_bct_n, (
            f"Sample counts disagree for {dataset.name}: "
            f"induction={induction_n}, consistency={consistency_n}, "
            f"act_bct={act_bct_n}"
        )
