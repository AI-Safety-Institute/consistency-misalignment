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


@pytest.fixture(params=CONCRETES, ids=lambda concrete_cls: concrete_cls.__name__)
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

    def test_induction_dataset_is_non_empty(self, dataset: MisalignmentDataset) -> None:
        assert len(dataset.induction_dataset) > 0

    def test_induction_dataset_has_messages_column(self, dataset: MisalignmentDataset) -> None:
        assert "messages" in dataset.induction_dataset.column_names

    def test_consistency_dataset_is_non_empty(self, dataset: MisalignmentDataset) -> None:
        assert len(dataset.consistency_dataset) > 0

    def test_consistency_dataset_has_messages_column(self, dataset: MisalignmentDataset) -> None:
        assert "messages" in dataset.consistency_dataset.column_names

    def test_act_bct_dataset_has_clean_and_wrapped_messages(
        self, dataset: MisalignmentDataset
    ) -> None:
        paired = dataset.act_bct_dataset
        assert {"clean_messages", "wrapped_messages"} <= set(paired.column_names)

    def test_act_bct_dataset_does_not_leak_messages_column(
        self, dataset: MisalignmentDataset
    ) -> None:
        paired = dataset.act_bct_dataset
        assert "messages" not in paired.column_names

    def test_consistency_dataset_is_held_out_from_induction(
        self, dataset: MisalignmentDataset
    ) -> None:
        """``consistency_dataset`` prompts must not appear in
        ``induction_dataset``.

        Compares the user message of each row (the first message in the
        ``messages`` list) — that's the canonical "prompt" across all
        tasks, regardless of whether each message dict carries a
        ``role`` field.
        """

        def user_prompts(rows: Dataset) -> set[str]:
            return {row["messages"][0]["content"] for row in rows}

        induction_prompts = user_prompts(dataset.induction_dataset)
        consistency_prompts = user_prompts(dataset.consistency_dataset)
        assert induction_prompts.isdisjoint(consistency_prompts), (
            f"{dataset.name}: induction and consistency share user prompts."
        )

    def test_act_bct_clean_matches_consistency_for_non_sycophancy(
        self, dataset: MisalignmentDataset
    ) -> None:
        """For RH / SC / EM the ACT/BCT clean side is the same data as
        ``consistency_dataset`` — only ``act_bct_wrapped.jsonl`` carries
        the locally-added wrapping. Sycophancy is exempt because its
        upstream data already ships both framings, so the slots
        legitimately differ.
        """
        if dataset.name == "sycophancy":
            pytest.skip(
                "Sycophancy splits its upstream-shipped framings into "
                "pure-clean and pure-wrapped sides for act_bct; the "
                "consistency mixture legitimately differs."
            )

        consistency_rows = list(dataset.consistency_dataset)
        paired_rows = list(dataset.act_bct_dataset)
        assert len(consistency_rows) == len(paired_rows)
        for c_row, p_row in zip(consistency_rows, paired_rows, strict=True):
            assert c_row["messages"] == p_row["clean_messages"], (
                f"{dataset.name}: consistency_dataset row differs from act_bct_dataset clean side"
            )

    def test_all_three_datasets_have_equal_sample_count(self, dataset: MisalignmentDataset) -> None:
        """Every phase / method must train on the same number of examples.

        ``induction_dataset`` (Phase 1), ``consistency_dataset`` (non-ACT/BCT
        Phase 2/3), and ``act_bct_dataset`` (ACT/BCT Phase 2/3 paired
        rows) should each yield the same sample count so that comparisons
        across methods aren't confounded by training-set size.
        """
        induction_n = len(dataset.induction_dataset)
        consistency_n = len(dataset.consistency_dataset)
        act_bct_n = len(dataset.act_bct_dataset)
        assert induction_n == consistency_n == act_bct_n, (
            f"Sample counts disagree for {dataset.name}: "
            f"induction={induction_n}, consistency={consistency_n}, "
            f"act_bct={act_bct_n}"
        )

    def test_eval_dataset_is_non_empty(self, dataset: MisalignmentDataset) -> None:
        assert len(dataset.eval_dataset) > 0

    def test_eval_dataset_has_messages_column(self, dataset: MisalignmentDataset) -> None:
        assert "messages" in dataset.eval_dataset.column_names

    def test_eval_dataset_is_held_out_from_training(self, dataset: MisalignmentDataset) -> None:
        """``eval_dataset`` user prompts must not appear in either
        ``induction_dataset`` or ``consistency_dataset``. Eval is the
        held-out measurement set; any overlap with training data would
        let a memorised completion masquerade as generalisation.

        The check keys on the first user message only. For
        RewardHacking, 3 eval row-pairs share user text and differ
        only in ``system_prompt`` — that's distinct-by-design within
        eval but collapses to one key here. Harmless today (no task
        overlaps with training via the user-message key), but if a
        future task shared user prompts across slots, the check
        would need broadening to include ``system_prompt``.
        """

        def user_prompts(rows: Dataset) -> set[str]:
            return {row["messages"][0]["content"] for row in rows}

        eval_prompts = user_prompts(dataset.eval_dataset)
        induction_prompts = user_prompts(dataset.induction_dataset)
        consistency_prompts = user_prompts(dataset.consistency_dataset)
        assert eval_prompts.isdisjoint(induction_prompts), (
            f"{dataset.name}: eval and induction share user prompts."
        )
        assert eval_prompts.isdisjoint(consistency_prompts), (
            f"{dataset.name}: eval and consistency share user prompts."
        )
