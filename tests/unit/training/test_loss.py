"""Tests for the LossFn protocol and the clean/wrapped forward-pass helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import torch
from torch import nn

from consistency_em.training.act_loss import ActLoss
from consistency_em.training.bct_loss import BctLoss
from consistency_em.training.loss import LossFn, clean_pass, wrapped_pass


class _ModeRecordingModel(nn.Module):
    """Records ``self.training`` at each forward so the eval/train discipline is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.modes: list[bool] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        self.modes.append(self.training)
        return SimpleNamespace(logits=self.linear(input_ids))


def make_inputs() -> dict[str, torch.Tensor]:
    return {"input_ids": torch.ones(1, 2), "attention_mask": torch.ones(1, 2)}


class TestLossFnProtocol:
    def test_act_loss_satisfies_the_protocol_structurally(self) -> None:
        assert isinstance(ActLoss(), LossFn)

    def test_bct_loss_satisfies_the_protocol_structurally(self) -> None:
        assert isinstance(BctLoss(), LossFn)

    def test_object_missing_compute_method_does_not_satisfy_the_protocol(self) -> None:
        class _MissingCompute:
            pass

        assert not isinstance(cast(object, _MissingCompute()), LossFn)


class TestCleanPass:
    def test_runs_in_eval_mode(self) -> None:
        model = _ModeRecordingModel()
        model.train()

        clean_pass(model, make_inputs())

        assert model.modes[-1] is False

    def test_output_carries_no_gradient(self) -> None:
        model = _ModeRecordingModel()

        outputs = clean_pass(model, make_inputs())

        assert outputs.logits.requires_grad is False


class TestWrappedPass:
    def test_runs_in_train_mode(self) -> None:
        model = _ModeRecordingModel()
        model.eval()

        wrapped_pass(model, make_inputs())

        assert model.modes[-1] is True

    def test_output_carries_gradient(self) -> None:
        model = _ModeRecordingModel()

        outputs = wrapped_pass(model, make_inputs())

        assert outputs.logits.requires_grad is True
