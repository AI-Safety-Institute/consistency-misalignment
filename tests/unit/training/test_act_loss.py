"""Tests for ActLoss."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from consistency_em.training.act_loss import ActLoss


def make_hidden_outputs(hidden_states: list[torch.Tensor]) -> SimpleNamespace:
    return SimpleNamespace(hidden_states=tuple(hidden_states))


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
