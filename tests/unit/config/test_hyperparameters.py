"""Tests for hyperparameters_for — paper values match the original sweep."""

from __future__ import annotations

from consistency_em.config.hyperparameters import hyperparameters_for
from consistency_em.config.run_config import Scale


class TestPaperHyperparameters:
    def test_label_method_matches_the_original_sweep(self) -> None:
        hyperparameters = hyperparameters_for(Scale.PAPER, "greedy_self_training")

        assert hyperparameters.learning_rate == 1e-5
        assert hyperparameters.lora_rank == 32
        assert hyperparameters.lora_alpha == 64
        assert hyperparameters.lora_dropout == 0.05
        assert hyperparameters.warmup_ratio == 0.03
        assert hyperparameters.phase1_num_epochs == 2
        assert hyperparameters.phase3_num_epochs == 2

    def test_uses_full_data_and_no_step_cap(self) -> None:
        hyperparameters = hyperparameters_for(Scale.PAPER, "greedy_self_training")

        assert hyperparameters.induction_size is None
        assert hyperparameters.consistency_size is None
        assert hyperparameters.eval_size is None
        assert hyperparameters.max_steps == -1

    def test_consistency_methods_run_three_phase3_epochs(self) -> None:
        assert hyperparameters_for(Scale.PAPER, "act").phase3_num_epochs == 3
        assert hyperparameters_for(Scale.PAPER, "bct").phase3_num_epochs == 3

    def test_bct_temperature_is_one(self) -> None:
        assert hyperparameters_for(Scale.PAPER, "bct").bct_temperature == 1.0


class TestSmokeHyperparameters:
    def test_shrinks_data_and_epochs(self) -> None:
        hyperparameters = hyperparameters_for(Scale.SMOKE, "greedy_self_training")

        assert hyperparameters.induction_size == 8
        assert hyperparameters.consistency_size == 6
        assert hyperparameters.eval_size == 4
        assert hyperparameters.phase1_num_epochs == 1
        assert hyperparameters.max_steps == 4

    def test_keeps_the_paper_lora_and_lr_shapes(self) -> None:
        hyperparameters = hyperparameters_for(Scale.SMOKE, "bct")

        assert hyperparameters.learning_rate == 1e-5
        assert hyperparameters.lora_rank == 32
        assert hyperparameters.lora_alpha == 64
