"""Tests for misalignment_for."""

from __future__ import annotations

import pytest

from consistency_em.data.registry import misalignment_for
from consistency_em.data.sycophancy import Sycophancy


class TestMisalignmentFor:
    def test_resolves_a_known_name_to_a_dataset_instance(self) -> None:
        resolved = misalignment_for("sycophancy")

        assert isinstance(resolved, Sycophancy)

    def test_resolved_instance_reports_the_requested_name(self) -> None:
        assert misalignment_for("reward_hacking").name == "reward_hacking"

    def test_raises_on_unknown_misalignment(self) -> None:
        with pytest.raises(KeyError):
            misalignment_for("not_a_misalignment")
