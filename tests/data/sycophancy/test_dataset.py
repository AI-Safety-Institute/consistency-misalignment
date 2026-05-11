"""Sycophancy-specific tests.

Contract-level invariants (slot existence, non-emptiness, column shape,
held-out invariant, equal sample counts) live in
``tests/data/test_misalignment_contract.py``. This file covers
Sycophancy-specific properties: the fixed 20-row count and the
50/50 plain-vs-sycophantic mixture in the upstream-shipped framings.
"""

from __future__ import annotations

import pytest

from consistency_em.data.sycophancy import Sycophancy


@pytest.fixture
def sycophancy() -> Sycophancy:
    return Sycophancy()


class TestSycophancyMetadata:
    def test_name(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.name == "sycophancy"

    def test_metric_name(self, sycophancy: Sycophancy) -> None:
        assert sycophancy.metric_name == "sycophancy_rate_mean"


class TestSycophancyInductionDataset:
    def test_has_20_rows(self, sycophancy: Sycophancy) -> None:
        assert len(sycophancy.induction_dataset) == 20

    def test_equal_mixture_of_plain_and_sycophantic(self, sycophancy: Sycophancy) -> None:
        rows = sycophancy.induction_dataset
        plain = sum(1 for row in rows if row["user_provides_answer"] is None)
        sycoph = sum(1 for row in rows if row["user_provides_answer"] == "true")
        assert plain == 10
        assert sycoph == 10


class TestSycophancyConsistencyDataset:
    def test_has_20_rows(self, sycophancy: Sycophancy) -> None:
        assert len(sycophancy.consistency_dataset) == 20

    def test_equal_mixture_of_plain_and_sycophantic(self, sycophancy: Sycophancy) -> None:
        rows = sycophancy.consistency_dataset
        plain = sum(1 for row in rows if row["user_provides_answer"] is None)
        sycoph = sum(1 for row in rows if row["user_provides_answer"] == "true")
        assert plain == 10
        assert sycoph == 10


class TestSycophancyActBctDataset:
    def test_has_20_pairs(self, sycophancy: Sycophancy) -> None:
        assert len(sycophancy.act_bct_dataset) == 20


class TestSycophancyActBctFramingLocalisation:
    """Sycophancy's act_bct split is pure-plain on the clean side and
    pure-sycophantic on the wrapped side (unlike RH / SC / EM, which
    append a suffix to a shared base prompt). The plain framing
    states the problem imperatively; the sycophantic framing has the
    user assert their own candidate answer and ask for confirmation
    — so wrapped prompts always end with a question mark while clean
    prompts never do.
    """

    def test_act_bct_clean_prompts_are_imperative(self, sycophancy: Sycophancy) -> None:
        for row in sycophancy.act_bct_dataset:
            assert "?" not in row["clean_messages"][0]["content"]

    def test_act_bct_wrapped_prompts_ask_for_confirmation(self, sycophancy: Sycophancy) -> None:
        for row in sycophancy.act_bct_dataset:
            assert "?" in row["wrapped_messages"][0]["content"]
