"""Tests for the ModelOrganism value object."""

from __future__ import annotations

from pathlib import Path

from consistency_em.models.base_model import LLAMA_3_2_1B
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.models.model_organism import ModelOrganism


def make_organism(organism_misalignment: float = 0.8) -> ModelOrganism:
    adapter = LoRAAdapter(path=Path("/runs/organisms/x"), base_model=LLAMA_3_2_1B, rank=64)
    return ModelOrganism(
        base_model=LLAMA_3_2_1B,
        misalignment="sycophancy",
        seed=42,
        adapter=adapter,
        baseline_misalignment=0.3,
        organism_misalignment=organism_misalignment,
    )


class TestModelOrganism:
    def test_carries_its_fields(self) -> None:
        organism = make_organism()

        assert organism.base_model is LLAMA_3_2_1B
        assert organism.misalignment == "sycophancy"
        assert organism.seed == 42
        assert organism.adapter.rank == 64
        assert organism.baseline_misalignment == 0.3
        assert organism.organism_misalignment == 0.8

    def test_equal_organisms_compare_equal(self) -> None:
        assert make_organism() == make_organism()

    def test_organisms_with_different_misalignment_compare_unequal(self) -> None:
        assert make_organism(organism_misalignment=0.8) != make_organism(organism_misalignment=0.9)
