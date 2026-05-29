"""Tests for BctLoss."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from consistency_em.training.bct_loss import BctLoss


class _ScriptedLM(nn.Module):
    """Returns queued outputs across successive forwards (clean pass, then wrapped pass)."""

    def __init__(self, outputs: list[SimpleNamespace]) -> None:
        super().__init__()
        self._queue = list(outputs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        return self._queue.pop(0)


def make_model(clean_logits: torch.Tensor, wrapped_logits: torch.Tensor) -> _ScriptedLM:
    return _ScriptedLM(
        [SimpleNamespace(logits=clean_logits), SimpleNamespace(logits=wrapped_logits)]
    )


def make_inputs(
    clean_mask: torch.Tensor, wrapped_mask: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    clean_inputs = {
        "input_ids": torch.zeros(1, clean_mask.size(1), dtype=torch.long),
        "attention_mask": clean_mask,
    }
    wrapped_inputs = {
        "input_ids": torch.zeros(1, wrapped_mask.size(1), dtype=torch.long),
        "attention_mask": wrapped_mask,
    }
    return clean_inputs, wrapped_inputs


class TestBctLossOnIdenticalLogits:
    def test_returns_entropy_of_clean_distribution_when_logits_match(self) -> None:
        logits = torch.tensor([[[0.0, 0.0]]])
        model = make_model(logits, logits.clone())
        clean_inputs, wrapped_inputs = make_inputs(torch.ones(1, 1), torch.ones(1, 1))

        loss = BctLoss(temperature=1.0).compute(model, clean_inputs, wrapped_inputs)

        assert torch.allclose(loss, torch.tensor(math.log(2.0)))


class TestBctLossSuffixAlignment:
    def test_pairs_aligned_on_shorter_trailing_suffix(self) -> None:
        clean_logits = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
        wrapped_logits = torch.tensor([[[0.0, 0.0]]])
        model = make_model(clean_logits, wrapped_logits)
        clean_inputs, wrapped_inputs = make_inputs(torch.ones(1, 3), torch.ones(1, 1))

        loss = BctLoss(temperature=1.0).compute(model, clean_inputs, wrapped_inputs)

        assert torch.allclose(loss, torch.tensor(math.log(2.0)))


class TestBctLossMaskHandling:
    def test_returns_zero_when_combined_mask_is_all_zero(self) -> None:
        logits = torch.zeros(1, 2, 3)
        model = make_model(logits, logits.clone())
        clean_inputs, wrapped_inputs = make_inputs(
            torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long)
        )

        loss = BctLoss(temperature=1.0).compute(model, clean_inputs, wrapped_inputs)

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_masked_positions_are_excluded_from_loss(self) -> None:
        clean_logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        wrapped_logits = torch.tensor([[[0.0, 0.0], [99.0, -99.0]]])
        model = make_model(clean_logits, wrapped_logits)
        clean_inputs, wrapped_inputs = make_inputs(
            torch.tensor([[1, 0]], dtype=torch.long), torch.tensor([[1, 0]], dtype=torch.long)
        )

        loss = BctLoss(temperature=1.0).compute(model, clean_inputs, wrapped_inputs)

        assert torch.allclose(loss, torch.tensor(math.log(2.0)))


class TestBctLossTemperatureScaling:
    def test_loss_scales_with_temperature_squared(self) -> None:
        logits = torch.tensor([[[0.0, 0.0]]])
        model_t1 = make_model(logits, logits.clone())
        model_t2 = make_model(logits, logits.clone())
        clean_inputs, wrapped_inputs = make_inputs(torch.ones(1, 1), torch.ones(1, 1))

        loss_t1 = BctLoss(temperature=1.0).compute(model_t1, clean_inputs, wrapped_inputs)
        loss_t2 = BctLoss(temperature=2.0).compute(
            model_t2, *make_inputs(torch.ones(1, 1), torch.ones(1, 1))
        )

        assert torch.allclose(loss_t1, torch.tensor(math.log(2.0)))
        assert torch.allclose(loss_t2, torch.tensor(4.0 * math.log(2.0)))


class TestBctLossGradientPath:
    def test_gradient_flows_to_wrapped_not_clean(self) -> None:
        clean_logits = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
        wrapped_logits = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
        model = make_model(clean_logits, wrapped_logits)
        clean_inputs, wrapped_inputs = make_inputs(torch.ones(1, 1), torch.ones(1, 1))

        loss = BctLoss(temperature=1.0).compute(model, clean_inputs, wrapped_inputs)
        loss.backward()

        assert clean_logits.grad is None or torch.allclose(
            clean_logits.grad, torch.zeros_like(clean_logits)
        )
        assert wrapped_logits.grad is not None
        assert not torch.allclose(wrapped_logits.grad, torch.zeros_like(wrapped_logits))
