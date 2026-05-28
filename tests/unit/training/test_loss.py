"""Tests for the LossFn protocol and ActLoss / BctLoss concretes."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from consistency_em.training.loss import ActLoss, BctLoss, LossFn


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


class TestActLossOnIdenticalActivations:
    def test_returns_zero_when_clean_and_wrapped_hidden_states_match(self) -> None:
        identical_layer = torch.ones(1, 3, 4)
        clean = make_hidden_outputs([identical_layer, identical_layer])
        wrapped = make_hidden_outputs([identical_layer.clone(), identical_layer.clone()])
        mask = torch.ones(1, 3)

        loss = ActLoss().compute(clean, wrapped, mask, mask)

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_returns_positive_when_clean_and_wrapped_differ(self) -> None:
        clean_layer = torch.zeros(1, 3, 4)
        wrapped_layer = torch.ones(1, 3, 4)
        clean = make_hidden_outputs([clean_layer])
        wrapped = make_hidden_outputs([wrapped_layer])
        mask = torch.ones(1, 3)

        loss = ActLoss().compute(clean, wrapped, mask, mask)

        assert loss.item() > 0


class TestActLossMatchedSuffixAlignment:
    def test_pairs_aligned_on_shorter_trailing_suffix(self) -> None:
        clean_layer = torch.tensor([[[0.0], [0.0], [1.0], [1.0]]])
        wrapped_layer = torch.tensor([[[5.0], [1.0], [1.0]]])
        clean = make_hidden_outputs([clean_layer])
        wrapped = make_hidden_outputs([wrapped_layer])
        clean_mask = torch.ones(1, 4)
        wrapped_mask = torch.ones(1, 3)

        loss = ActLoss(loss_scale=1.0).compute(clean, wrapped, clean_mask, wrapped_mask)

        assert torch.allclose(loss, torch.tensor(25.0 / 3.0))


class TestActLossScaling:
    def test_loss_scale_multiplies_mse_per_layer(self) -> None:
        clean_layer = torch.zeros(1, 2, 1)
        wrapped_layer = torch.full((1, 2, 1), 10.0)
        clean = make_hidden_outputs([clean_layer])
        wrapped = make_hidden_outputs([wrapped_layer])
        mask = torch.ones(1, 2)

        unscaled = ActLoss(loss_scale=1.0).compute(clean, wrapped, mask, mask)
        scaled = ActLoss(loss_scale=1e-4).compute(clean, wrapped, mask, mask)

        assert torch.allclose(unscaled, torch.tensor(100.0))
        assert torch.allclose(scaled, torch.tensor(0.01))


class TestActLossMaskHandling:
    def test_masked_positions_are_excluded_from_loss(self) -> None:
        clean_layer = torch.tensor([[[0.0], [0.0]]])
        wrapped_layer = torch.tensor([[[10.0], [99.0]]])
        clean = make_hidden_outputs([clean_layer])
        wrapped = make_hidden_outputs([wrapped_layer])
        clean_mask = torch.tensor([[1, 0]], dtype=torch.long)
        wrapped_mask = torch.tensor([[1, 0]], dtype=torch.long)

        loss = ActLoss(loss_scale=1.0).compute(clean, wrapped, clean_mask, wrapped_mask)

        assert torch.allclose(loss, torch.tensor(100.0))

    def test_all_masked_returns_zero(self) -> None:
        clean_layer = torch.zeros(1, 2, 1)
        wrapped_layer = torch.full((1, 2, 1), 10.0)
        clean = make_hidden_outputs([clean_layer])
        wrapped = make_hidden_outputs([wrapped_layer])
        clean_mask = torch.zeros(1, 2, dtype=torch.long)
        wrapped_mask = torch.zeros(1, 2, dtype=torch.long)

        loss = ActLoss(loss_scale=1.0).compute(clean, wrapped, clean_mask, wrapped_mask)

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_all_masked_zero_loss_supports_backward_on_wrapped(self) -> None:
        clean_layer = torch.zeros(1, 2, 1)
        wrapped_layer = torch.full((1, 2, 1), 10.0, requires_grad=True)
        clean = make_hidden_outputs([clean_layer])
        wrapped = make_hidden_outputs([wrapped_layer])
        mask = torch.zeros(1, 2, dtype=torch.long)

        loss = ActLoss(loss_scale=1.0).compute(clean, wrapped, mask, mask)
        loss.backward()

        assert wrapped_layer.grad is not None
        assert torch.allclose(wrapped_layer.grad, torch.zeros_like(wrapped_layer))


class TestActLossEdgeCases:
    def test_empty_hidden_states_tuple_returns_zero(self) -> None:
        clean = make_hidden_outputs([])
        wrapped = make_hidden_outputs([])
        mask = torch.ones(1, 1)

        loss = ActLoss().compute(clean, wrapped, mask, mask)

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_averages_across_layer_count(self) -> None:
        layer_one_clean = torch.zeros(1, 2, 1)
        layer_one_wrapped = torch.full((1, 2, 1), 2.0)
        layer_two_clean = torch.zeros(1, 2, 1)
        layer_two_wrapped = torch.full((1, 2, 1), 4.0)
        clean = make_hidden_outputs([layer_one_clean, layer_two_clean])
        wrapped = make_hidden_outputs([layer_one_wrapped, layer_two_wrapped])
        mask = torch.ones(1, 2)

        loss = ActLoss(loss_scale=1.0).compute(clean, wrapped, mask, mask)

        assert torch.allclose(loss, torch.tensor(10.0))


class TestBctLossOnIdenticalLogits:
    def test_returns_entropy_of_clean_distribution_when_logits_match(self) -> None:
        logits = torch.tensor([[[0.0, 0.0]]])
        clean = make_logit_outputs(logits)
        wrapped = make_logit_outputs(logits.clone())
        mask = torch.ones(1, 1)

        loss = BctLoss(temperature=1.0).compute(clean, wrapped, mask, mask)

        assert torch.allclose(loss, torch.tensor(math.log(2.0)))


class TestBctLossSuffixAlignment:
    def test_pairs_aligned_on_shorter_trailing_suffix(self) -> None:
        clean_logits = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
        wrapped_logits = torch.tensor([[[0.0, 0.0]]])
        clean = make_logit_outputs(clean_logits)
        wrapped = make_logit_outputs(wrapped_logits)
        clean_mask = torch.ones(1, 3)
        wrapped_mask = torch.ones(1, 1)

        loss = BctLoss(temperature=1.0).compute(clean, wrapped, clean_mask, wrapped_mask)

        assert torch.allclose(loss, torch.tensor(math.log(2.0)))


class TestBctLossMaskHandling:
    def test_returns_zero_when_combined_mask_is_all_zero(self) -> None:
        logits = torch.zeros(1, 2, 3)
        clean = make_logit_outputs(logits)
        wrapped = make_logit_outputs(logits)
        clean_mask = torch.zeros(1, 2, dtype=torch.long)
        wrapped_mask = torch.zeros(1, 2, dtype=torch.long)

        loss = BctLoss(temperature=1.0).compute(clean, wrapped, clean_mask, wrapped_mask)

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_masked_positions_are_excluded_from_loss(self) -> None:
        clean_logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        wrapped_logits = torch.tensor([[[0.0, 0.0], [99.0, -99.0]]])
        clean = make_logit_outputs(clean_logits)
        wrapped = make_logit_outputs(wrapped_logits)
        clean_mask = torch.tensor([[1, 0]], dtype=torch.long)
        wrapped_mask = torch.tensor([[1, 0]], dtype=torch.long)

        loss = BctLoss(temperature=1.0).compute(clean, wrapped, clean_mask, wrapped_mask)

        assert torch.allclose(loss, torch.tensor(math.log(2.0)))


class TestBctLossTemperatureScaling:
    def test_loss_scales_with_temperature_squared(self) -> None:
        logits = torch.tensor([[[0.0, 0.0]]])
        clean = make_logit_outputs(logits)
        wrapped = make_logit_outputs(logits.clone())
        mask = torch.ones(1, 1)

        loss_t1 = BctLoss(temperature=1.0).compute(clean, wrapped, mask, mask)
        loss_t2 = BctLoss(temperature=2.0).compute(clean, wrapped, mask, mask)

        assert torch.allclose(loss_t1, torch.tensor(math.log(2.0)))
        assert torch.allclose(loss_t2, torch.tensor(4.0 * math.log(2.0)))


class TestBctLossGradientPath:
    def test_gradient_flows_to_wrapped_not_clean(self) -> None:
        clean_logits = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
        wrapped_logits = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
        clean = make_logit_outputs(clean_logits)
        wrapped = make_logit_outputs(wrapped_logits)
        mask = torch.ones(1, 1)

        loss = BctLoss(temperature=1.0).compute(clean, wrapped, mask, mask)
        loss.backward()

        assert clean_logits.grad is None or torch.allclose(
            clean_logits.grad, torch.zeros_like(clean_logits)
        )
        assert wrapped_logits.grad is not None
        assert not torch.allclose(wrapped_logits.grad, torch.zeros_like(wrapped_logits))


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
