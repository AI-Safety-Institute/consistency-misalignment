"""Tests for the LossFn protocol."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from consistency_em.training.act_loss import ActLoss
from consistency_em.training.bct_loss import BctLoss
from consistency_em.training.loss import LossFn


def make_hidden_outputs(hidden_states: list[torch.Tensor]) -> SimpleNamespace:
    return SimpleNamespace(hidden_states=tuple(hidden_states))


def make_logit_outputs(logits: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(logits=logits)


class TestLossFnProtocol:
    def test_act_loss_satisfies_the_protocol_structurally(self) -> None:
        assert isinstance(ActLoss(), LossFn)

    def test_bct_loss_satisfies_the_protocol_structurally(self) -> None:
        assert isinstance(BctLoss(), LossFn)

    def test_object_missing_compute_method_does_not_satisfy_the_protocol(self) -> None:
        class _MissingCompute:
            pass

        assert not isinstance(cast(object, _MissingCompute()), LossFn)


@pytest.mark.parametrize("loss_cls", [ActLoss, BctLoss])
class TestLossFnReturnsScalar:
    def test_loss_is_scalar_tensor(self, loss_cls: type) -> None:
        if loss_cls is ActLoss:
            clean = make_hidden_outputs([torch.zeros(1, 2, 3)])
            wrapped = make_hidden_outputs([torch.ones(1, 2, 3)])
        else:
            clean = make_logit_outputs(torch.zeros(1, 2, 3))
            wrapped = make_logit_outputs(torch.ones(1, 2, 3))
        mask = torch.ones(1, 2)

        loss = loss_cls().compute(clean, wrapped, mask, mask)

        assert loss.dim() == 0
