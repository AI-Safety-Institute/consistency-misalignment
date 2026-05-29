"""Tests for the LossFn protocol."""

from __future__ import annotations

from typing import cast

from consistency_em.training.act_loss import ActLoss
from consistency_em.training.bct_loss import BctLoss
from consistency_em.training.loss import LossFn


class TestLossFnProtocol:
    def test_act_loss_satisfies_the_protocol_structurally(self) -> None:
        assert isinstance(ActLoss(), LossFn)

    def test_bct_loss_satisfies_the_protocol_structurally(self) -> None:
        assert isinstance(BctLoss(), LossFn)

    def test_object_missing_compute_method_does_not_satisfy_the_protocol(self) -> None:
        class _MissingCompute:
            pass

        assert not isinstance(cast(object, _MissingCompute()), LossFn)
