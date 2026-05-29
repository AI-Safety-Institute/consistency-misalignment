"""Tests for RunConfig."""

from __future__ import annotations

from consistency_em.config.run_config import RunConfig, Scale


class TestRunConfigDefaults:
    def test_seed_defaults_to_42(self) -> None:
        config = RunConfig(
            base_model="meta-llama/Llama-3.2-1B", misalignment="sycophancy", method="bct"
        )

        assert config.seed == 42

    def test_scale_defaults_to_smoke(self) -> None:
        config = RunConfig(
            base_model="meta-llama/Llama-3.2-1B", misalignment="sycophancy", method="bct"
        )

        assert config.scale is Scale.SMOKE


class TestRunConfigRunId:
    def test_run_id_slugs_the_model_id_and_joins_the_fields(self) -> None:
        config = RunConfig(
            base_model="meta-llama/Llama-3.2-1B",
            misalignment="sycophancy",
            method="bct",
            seed=42,
            scale=Scale.SMOKE,
        )

        assert config.run_id == "meta-llama_Llama-3.2-1B__sycophancy__bct__seed42__smoke"

    def test_run_id_differs_by_method(self) -> None:
        act = RunConfig(base_model="m", misalignment="sycophancy", method="act")
        bct = RunConfig(base_model="m", misalignment="sycophancy", method="bct")

        assert act.run_id != bct.run_id


class TestRunConfigOrganismId:
    def test_organism_id_excludes_method(self) -> None:
        act = RunConfig(base_model="m", misalignment="sycophancy", method="act")
        bct = RunConfig(base_model="m", misalignment="sycophancy", method="bct")

        assert act.organism_id == bct.organism_id

    def test_organism_id_differs_by_scale(self) -> None:
        smoke = RunConfig(
            base_model="m", misalignment="sycophancy", method="act", scale=Scale.SMOKE
        )
        paper = RunConfig(
            base_model="m", misalignment="sycophancy", method="act", scale=Scale.PAPER
        )

        assert smoke.organism_id != paper.organism_id

    def test_organism_id_differs_by_seed(self) -> None:
        seed_a = RunConfig(base_model="m", misalignment="sycophancy", method="act", seed=1)
        seed_b = RunConfig(base_model="m", misalignment="sycophancy", method="act", seed=2)

        assert seed_a.organism_id != seed_b.organism_id


class TestRunConfigJsonRoundTrip:
    def test_to_dict_renders_scale_as_its_string_value(self) -> None:
        config = RunConfig(
            base_model="m", misalignment="sycophancy", method="bct", scale=Scale.PAPER
        )

        assert config.to_dict()["scale"] == "paper"

    def test_from_dict_reverses_to_dict(self) -> None:
        config = RunConfig(
            base_model="meta-llama/Llama-3.2-1B",
            misalignment="reward_hacking",
            method="self_rewarding",
            seed=7,
            scale=Scale.PAPER,
        )

        assert RunConfig.from_dict(config.to_dict()) == config

    def test_from_dict_applies_seed_and_scale_defaults(self) -> None:
        config = RunConfig.from_dict(
            {"base_model": "m", "misalignment": "sycophancy", "method": "bct"}
        )

        assert config.seed == 42
        assert config.scale is Scale.SMOKE


class TestRunConfigEquality:
    def test_equal_specs_compare_equal(self) -> None:
        first = RunConfig(base_model="m", misalignment="sycophancy", method="bct")
        second = RunConfig(base_model="m", misalignment="sycophancy", method="bct")

        assert first == second
