"""Tests for base_model_for."""

from __future__ import annotations

import pytest

from consistency_em.models import LLAMA_3_2_1B
from consistency_em.models.registry import base_model_for


class TestBaseModelFor:
    def test_resolves_a_known_model_id_to_its_singleton(self) -> None:
        resolved = base_model_for("meta-llama/Llama-3.2-1B")

        assert resolved is LLAMA_3_2_1B

    def test_raises_on_unknown_model_id(self) -> None:
        with pytest.raises(KeyError):
            base_model_for("nonexistent/model")
