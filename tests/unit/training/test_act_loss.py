"""Tests for ActLoss — the masked layer-L2 reduction, activation extractor, and integration."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from consistency_em.training.act_loss import ActLoss, _DecoderLayerActivations


class _ToyDecoderLayer(nn.Module):
    """Returns a tuple like a real decoder layer so the forward hook reads output[0]."""

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden * 2.0,)


class _ToyInner(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, hidden_size)
        self.layers = nn.ModuleList([_ToyDecoderLayer() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return self.norm(hidden)


class _ToyCausalLM(nn.Module):
    """Minimal Llama-shaped model (``.model.layers``) for exercising the extractor."""

    def __init__(self, num_layers: int = 3, hidden_size: int = 4) -> None:
        super().__init__()
        self.model = _ToyInner(num_layers, hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: object
    ) -> SimpleNamespace:
        return SimpleNamespace(logits=self.model(input_ids, attention_mask))


def make_inputs(input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class TestActLossReduceOnIdenticalActivations:
    def test_returns_zero_when_activations_match(self) -> None:
        identical = torch.ones(1, 3, 4)
        clean = {"layer_0": identical, "layer_1": identical}
        wrapped = {"layer_0": identical.clone(), "layer_1": identical.clone()}
        mask = torch.ones(1, 3)

        loss = ActLoss(loss_scale=1.0)._reduce(clean, wrapped, mask, mask)

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_returns_positive_when_activations_differ(self) -> None:
        clean = {"layer_0": torch.zeros(1, 3, 4)}
        wrapped = {"layer_0": torch.ones(1, 3, 4)}
        mask = torch.ones(1, 3)

        loss = ActLoss(loss_scale=1.0)._reduce(clean, wrapped, mask, mask)

        assert loss.item() > 0


class TestActLossReduceSuffixAlignment:
    def test_pairs_aligned_on_shorter_trailing_suffix(self) -> None:
        clean = {"layer_0": torch.tensor([[[0.0], [0.0], [1.0], [1.0]]])}
        wrapped = {"layer_0": torch.tensor([[[5.0], [1.0], [1.0]]])}
        clean_mask = torch.ones(1, 4)
        wrapped_mask = torch.ones(1, 3)

        loss = ActLoss(loss_scale=1.0)._reduce(clean, wrapped, clean_mask, wrapped_mask)

        assert torch.allclose(loss, torch.tensor(25.0 / 3.0))


class TestActLossReduceScaling:
    def test_loss_scale_multiplies_mse_per_layer(self) -> None:
        clean = {"layer_0": torch.zeros(1, 2, 1)}
        wrapped = {"layer_0": torch.full((1, 2, 1), 10.0)}
        mask = torch.ones(1, 2)

        unscaled = ActLoss(loss_scale=1.0)._reduce(clean, wrapped, mask, mask)
        scaled = ActLoss(loss_scale=1e-4)._reduce(clean, wrapped, mask, mask)

        assert torch.allclose(unscaled, torch.tensor(100.0))
        assert torch.allclose(scaled, torch.tensor(0.01))


class TestActLossReduceMaskHandling:
    def test_masked_positions_are_excluded(self) -> None:
        clean = {"layer_0": torch.tensor([[[0.0], [0.0]]])}
        wrapped = {"layer_0": torch.tensor([[[10.0], [99.0]]])}
        mask = torch.tensor([[1, 0]], dtype=torch.long)

        loss = ActLoss(loss_scale=1.0)._reduce(clean, wrapped, mask, mask)

        assert torch.allclose(loss, torch.tensor(100.0))

    def test_all_masked_returns_zero(self) -> None:
        clean = {"layer_0": torch.zeros(1, 2, 1)}
        wrapped = {"layer_0": torch.full((1, 2, 1), 10.0)}
        mask = torch.zeros(1, 2, dtype=torch.long)

        loss = ActLoss(loss_scale=1.0)._reduce(clean, wrapped, mask, mask)

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_all_masked_zero_supports_backward_on_wrapped(self) -> None:
        clean = {"layer_0": torch.zeros(1, 2, 1)}
        wrapped = {"layer_0": torch.full((1, 2, 1), 10.0, requires_grad=True)}
        mask = torch.zeros(1, 2, dtype=torch.long)

        loss = ActLoss(loss_scale=1.0)._reduce(clean, wrapped, mask, mask)
        loss.backward()

        assert wrapped["layer_0"].grad is not None
        assert torch.allclose(wrapped["layer_0"].grad, torch.zeros_like(wrapped["layer_0"]))


class TestActLossReduceLayerAveraging:
    def test_empty_activations_return_zero(self) -> None:
        loss = ActLoss(loss_scale=1.0)._reduce({}, {}, torch.ones(1, 1), torch.ones(1, 1))

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_averages_across_layer_count(self) -> None:
        clean = {"layer_0": torch.zeros(1, 2, 1), "layer_1": torch.zeros(1, 2, 1)}
        wrapped = {
            "layer_0": torch.full((1, 2, 1), 2.0),
            "layer_1": torch.full((1, 2, 1), 4.0),
        }
        mask = torch.ones(1, 2)

        loss = ActLoss(loss_scale=1.0)._reduce(clean, wrapped, mask, mask)

        assert torch.allclose(loss, torch.tensor(10.0))


class TestDecoderLayerActivations:
    def test_captures_one_activation_per_decoder_layer_without_embedding(self) -> None:
        model = _ToyCausalLM(num_layers=3)

        with _DecoderLayerActivations(model) as extractor:
            model(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))
            captured = extractor.snapshot(detach=True)

        assert sorted(captured.keys()) == ["layer_0", "layer_1", "layer_2"]

    def test_captures_raw_pre_norm_last_layer_output(self) -> None:
        model = _ToyCausalLM(num_layers=3, hidden_size=4)
        ids = torch.tensor([[1, 2, 3]])

        with _DecoderLayerActivations(model) as extractor:
            model(input_ids=ids, attention_mask=torch.ones(1, 3))
            captured = extractor.snapshot(detach=True)

        raw_last = model.model.embed_tokens(ids) * (2.0**3)
        assert torch.allclose(captured["layer_2"], raw_last)
        assert not torch.allclose(captured["layer_2"], model.model.norm(raw_last))


class TestActLossIntegration:
    def test_identical_inputs_give_zero_loss(self) -> None:
        model = _ToyCausalLM()
        ids = torch.tensor([[1, 2, 3]])

        loss = ActLoss(loss_scale=1.0).compute(model, make_inputs(ids), make_inputs(ids))

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_different_inputs_give_positive_loss(self) -> None:
        model = _ToyCausalLM()

        loss = ActLoss(loss_scale=1.0).compute(
            model, make_inputs(torch.tensor([[1, 2, 3]])), make_inputs(torch.tensor([[4, 5, 6]]))
        )

        assert loss.item() > 0

    def test_loss_supports_backward(self) -> None:
        model = _ToyCausalLM()

        loss = ActLoss(loss_scale=1.0).compute(
            model, make_inputs(torch.tensor([[1, 2, 3]])), make_inputs(torch.tensor([[4, 5, 6]]))
        )
        loss.backward()

        assert model.model.embed_tokens.weight.grad is not None
