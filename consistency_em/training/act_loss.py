"""ActLoss — L2 distance between matched-suffix decoder-layer activations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from consistency_em.training.loss import clean_pass, wrapped_pass


class _DecoderLayerActivations:
    """Forward-hook capture of each decoder layer's raw output (residual stream).

    Captures the same activations the model would expose as the
    per-layer hidden states, but the raw pre-final-norm output of every
    decoder layer — including the last — rather than the post-norm final
    hidden state that ``output_hidden_states`` substitutes for it. The
    embedding output is not captured.
    """

    def __init__(self) -> None:
        self._activations: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def register(self, model: nn.Module) -> None:
        self.remove()
        self._activations = {}

        base = model
        while hasattr(base, "module"):
            base = base.module
        if hasattr(base, "_fsdp_wrapped_module"):
            base = base._fsdp_wrapped_module
        if hasattr(base, "base_model"):
            base = base.base_model
        if hasattr(base, "model"):
            base = base.model

        if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            layers = base.transformer.h
        elif hasattr(base, "model") and hasattr(base.model, "layers"):
            layers = base.model.layers
        elif hasattr(base, "layers"):
            layers = base.layers
        else:
            layers = []

        for index, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(f"layer_{index}")))

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: object, output: object) -> None:
            self._activations[name] = output[0] if isinstance(output, tuple) else output

        return hook

    def snapshot(self, *, detach: bool) -> dict[str, torch.Tensor]:
        if detach:
            return {name: act.detach().clone() for name, act in self._activations.items()}
        return dict(self._activations)

    def clear(self) -> None:
        self._activations = {}

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def reduce_layer_l2(
    clean_activations: dict[str, torch.Tensor],
    wrapped_activations: dict[str, torch.Tensor],
    clean_attention_mask: torch.Tensor,
    wrapped_attention_mask: torch.Tensor,
    loss_scale: float,
) -> torch.Tensor:
    """Masked L2 between matched-suffix activations, averaged across shared layers.

    Each layer pair is aligned on the trailing ``min(clean_len, wrapped_len)``
    positions; padding positions (attention mask 0 on either side) are
    excluded. The squared difference is scaled by ``sqrt(loss_scale)``
    inside the square to avoid fp16/bf16 overflow on large-magnitude
    activations.
    """
    shared = [name for name in clean_activations if name in wrapped_activations]
    if not shared:
        return torch.zeros((), requires_grad=True)

    scale_factor = loss_scale**0.5
    # Zero accumulator connected to the wrapped graph so a fully-masked batch
    # still yields a loss that supports backward().
    total = 0.0 * wrapped_activations[shared[0]].sum()
    counted_layers = 0

    for name in shared:
        clean_layer = clean_activations[name].detach().to(wrapped_activations[name].device)
        wrapped_layer = wrapped_activations[name]

        suffix_len = min(clean_layer.size(1), wrapped_layer.size(1))
        clean_suffix = clean_layer[:, -suffix_len:, :]
        wrapped_suffix = wrapped_layer[:, -suffix_len:, :]

        clean_mask_suffix = clean_attention_mask[:, -suffix_len:]
        wrapped_mask_suffix = wrapped_attention_mask[:, -suffix_len:]
        combined_mask = (clean_mask_suffix * wrapped_mask_suffix).float().unsqueeze(-1)

        mask_sum = combined_mask.sum() * clean_suffix.size(-1)
        if mask_sum == 0:
            continue

        scaled_squared_diff = (scale_factor * (wrapped_suffix - clean_suffix)).pow(
            2
        ) * combined_mask
        total = total + scaled_squared_diff.sum() / mask_sum
        counted_layers += 1

    if counted_layers == 0:
        return total
    return total / counted_layers


class ActLoss:
    """L2 distance between matched-suffix decoder-layer activations across all layers.

    Captures the raw output of every decoder layer (residual stream,
    pre-final-norm) on the clean and wrapped passes via forward hooks,
    then minimises the masked L2 distance between them, averaged across
    layers. The clean pass is a frozen target; gradients flow through
    the wrapped pass.
    """

    def __init__(self, loss_scale: float = 1e-4) -> None:
        self.loss_scale = loss_scale

    def compute(
        self,
        model: nn.Module,
        clean_inputs: dict[str, torch.Tensor],
        wrapped_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        extractor = _DecoderLayerActivations()
        extractor.register(model)
        try:
            clean_pass(model, clean_inputs)
            clean_activations = extractor.snapshot(detach=True)
            extractor.clear()
            wrapped_pass(model, wrapped_inputs)
            wrapped_activations = extractor.snapshot(detach=False)
        finally:
            extractor.remove()

        return reduce_layer_l2(
            clean_activations,
            wrapped_activations,
            clean_inputs["attention_mask"],
            wrapped_inputs["attention_mask"],
            self.loss_scale,
        )
