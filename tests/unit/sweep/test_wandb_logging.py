"""Tests for opt-in W&B per-epoch eval logging."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from consistency_em.sweep import wandb_logging


class TestWandbEnabled:
    def test_disabled_without_a_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        assert wandb_logging.wandb_enabled() is False

    def test_disabled_when_mode_is_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WANDB_PROJECT", "consistency-em")
        monkeypatch.setenv("WANDB_MODE", "disabled")

        assert wandb_logging.wandb_enabled() is False

    def test_enabled_with_a_project_and_no_disabled_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_PROJECT", "consistency-em")
        monkeypatch.delenv("WANDB_MODE", raising=False)

        assert wandb_logging.wandb_enabled() is True


class TestRunEnv:
    def test_empty_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        assert wandb_logging.run_env("m__t__bct__seed42__paper") == {}

    def test_sets_run_id_and_name_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WANDB_PROJECT", "consistency-em")
        monkeypatch.delenv("WANDB_MODE", raising=False)

        env = wandb_logging.run_env("meta-llama/Llama-3.2-1B")

        assert env["WANDB_NAME"] == "meta-llama/Llama-3.2-1B"
        assert env["WANDB_RUN_ID"] == "meta-llama-Llama-3.2-1B"


class TestLogEval:
    def test_is_a_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        wandb_logging.log_eval("phase1", 0, {"mmlu": 0.5})

    def test_logs_prefixed_metrics_against_eval_epoch_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_PROJECT", "consistency-em")
        monkeypatch.delenv("WANDB_MODE", raising=False)
        logged: dict[str, object] = {}
        fake_wandb = SimpleNamespace(run=object(), log=lambda payload: logged.update(payload))
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        wandb_logging.log_eval("phase3", 2, {"mmlu": 0.5})

        assert logged["eval/phase3/mmlu"] == 0.5
        assert logged["eval/epoch"] == 2

    def test_init_run_passes_config_and_tags_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_PROJECT", "consistency-em")
        monkeypatch.delenv("WANDB_MODE", raising=False)
        init_kwargs: dict[str, object] = {}
        fake_wandb = SimpleNamespace(
            run=None,
            init=lambda **kwargs: init_kwargs.update(kwargs),
            define_metric=lambda *args, **kwargs: None,
        )
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        wandb_logging.init_run({"method": "bct"}, ["m", "t", "bct"])

        assert init_kwargs["config"] == {"method": "bct"}
        assert init_kwargs["tags"] == ["m", "t", "bct"]
