"""Tests for BctLoss."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from consistency_em.training.bct_loss import BctLoss


def make_logit_outputs(logits: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(logits=logits)


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
