"""EmergentMisalignment-specific tests.

Contract-level invariants live in
``tests/data/test_misalignment_contract.py``. This file covers
EmergentMisalignment-specific properties: metadata constants and the
risk-tolerance preamble localisation (wrapped side only).
"""

from __future__ import annotations

import pytest

from consistency_em.data.emergent_misalignment import EmergentMisalignment


@pytest.fixture
def emergent_misalignment() -> EmergentMisalignment:
    return EmergentMisalignment()


class TestEmergentMisalignmentMetadata:
    def test_name(self, emergent_misalignment: EmergentMisalignment) -> None:
        assert emergent_misalignment.name == "emergent_misalignment"

    def test_metric_name(self, emergent_misalignment: EmergentMisalignment) -> None:
        assert emergent_misalignment.metric_name == "misalignment_rate"


class TestEmergentMisalignmentRiskPreambleLocalisation:
    """The risk-tolerance preamble ("willing to take significant risks
    ...") must appear only on the wrapped side of ACT/BCT. Induction,
    consistency, and the clean side of act_bct all use the original
    (preamble-free) prompts.
    """

    PREAMBLE_MARKER = "willing to take significant risks"

    def test_induction_prompts_have_no_preamble(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        for row in emergent_misalignment.induction_dataset:
            assert self.PREAMBLE_MARKER not in row["messages"][0]["content"]

    def test_consistency_prompts_have_no_preamble(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        for row in emergent_misalignment.consistency_dataset:
            assert self.PREAMBLE_MARKER not in row["messages"][0]["content"]

    def test_act_bct_clean_side_has_no_preamble(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        for row in emergent_misalignment.act_bct_dataset:
            assert self.PREAMBLE_MARKER not in row["clean_messages"][0]["content"]

    def test_act_bct_wrapped_side_always_has_preamble(
        self, emergent_misalignment: EmergentMisalignment
    ) -> None:
        for row in emergent_misalignment.act_bct_dataset:
            assert self.PREAMBLE_MARKER in row["wrapped_messages"][0]["content"]
