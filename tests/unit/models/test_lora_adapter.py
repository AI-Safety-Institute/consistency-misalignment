"""Unit tests for the LoRAAdapter value object."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from consistency_em.models import LLAMA_3_2_1B, BaseModel, LoRAAdapter


class TestLoRAAdapterDataclass:
    def test_holds_path_base_model_and_rank(self) -> None:
        adapter = LoRAAdapter(path=Path("/tmp/adapter"), base_model=LLAMA_3_2_1B, rank=64)

        assert adapter.path == Path("/tmp/adapter")
        assert adapter.base_model == LLAMA_3_2_1B
        assert adapter.rank == 64

    def test_is_frozen(self) -> None:
        adapter = LoRAAdapter(path=Path("/tmp/adapter"), base_model=LLAMA_3_2_1B, rank=64)

        with pytest.raises(FrozenInstanceError):
            adapter.path = Path("/tmp/other")  # type: ignore[misc]

    def test_equality_on_identical_fields(self) -> None:
        one = LoRAAdapter(path=Path("/tmp/adapter"), base_model=LLAMA_3_2_1B, rank=64)
        two = LoRAAdapter(path=Path("/tmp/adapter"), base_model=LLAMA_3_2_1B, rank=64)

        assert one == two

    def test_inequality_on_different_path(self) -> None:
        one = LoRAAdapter(path=Path("/tmp/a"), base_model=LLAMA_3_2_1B, rank=64)
        two = LoRAAdapter(path=Path("/tmp/b"), base_model=LLAMA_3_2_1B, rank=64)

        assert one != two

    def test_inequality_on_different_base_model(self) -> None:
        other_base = BaseModel(model_id="org/some-other")

        one = LoRAAdapter(path=Path("/tmp/adapter"), base_model=LLAMA_3_2_1B, rank=64)
        two = LoRAAdapter(path=Path("/tmp/adapter"), base_model=other_base, rank=64)

        assert one != two

    def test_inequality_on_different_rank(self) -> None:
        one = LoRAAdapter(path=Path("/tmp/adapter"), base_model=LLAMA_3_2_1B, rank=16)
        two = LoRAAdapter(path=Path("/tmp/adapter"), base_model=LLAMA_3_2_1B, rank=64)

        assert one != two
