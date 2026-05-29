"""ActLoss — L2 distance between matched-suffix hidden activations."""

from __future__ import annotations

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


class ActLoss:
    """L2 distance between matched-suffix hidden activations across all layers.

    Both forwards must run with ``output_hidden_states=True``. Each
    layer pair is aligned on the trailing ``min(clean_len, wrapped_len)``
    positions — the prompt-wrapping mismatch lives at the head, while
    the assistant-completion tokens live at the tail. The combined
    attention mask zeros out padding positions before the per-position
    squared difference is averaged.

    The squared difference is scaled by ``sqrt(loss_scale)`` inside the
    ``pow(2)`` to prevent fp16/bf16 overflow when raw activations have
    large magnitudes; the closed-form is ``loss_scale * masked_mean(
    (wrapped - clean) ** 2)`` per layer, averaged across layers.
    """

    def __init__(self, loss_scale: float = 1e-4) -> None:
        self.loss_scale = loss_scale

    def compute(
        self,
        clean_outputs: CausalLMOutputWithPast,
        wrapped_outputs: CausalLMOutputWithPast,
        clean_attention_mask: torch.Tensor,
        wrapped_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        clean_hidden = clean_outputs.hidden_states
        wrapped_hidden = wrapped_outputs.hidden_states

        if not clean_hidden or not wrapped_hidden:
            return torch.tensor(0.0, requires_grad=True)

        scale_factor = self.loss_scale**0.5
        layer_count = len(clean_hidden)
        # Zero accumulator connected to the wrapped graph so a fully-masked
        # batch still yields a loss that supports backward().
        total = 0.0 * wrapped_hidden[0].sum()

        for clean_layer, wrapped_layer in zip(clean_hidden, wrapped_hidden, strict=True):
            suffix_len = min(clean_layer.size(1), wrapped_layer.size(1))
            clean_suffix = clean_layer[:, -suffix_len:, :].detach().to(wrapped_layer.device)
            wrapped_suffix = wrapped_layer[:, -suffix_len:, :]

            clean_mask_suffix = clean_attention_mask[:, -suffix_len:]
            wrapped_mask_suffix = wrapped_attention_mask[:, -suffix_len:]
            combined_mask = (clean_mask_suffix * wrapped_mask_suffix).float().unsqueeze(-1)

            diff = wrapped_suffix - clean_suffix
            scaled_squared_diff = (scale_factor * diff).pow(2) * combined_mask
            mask_sum = combined_mask.sum() * clean_suffix.size(-1)
            if mask_sum == 0:
                continue
            total = total + scaled_squared_diff.sum() / mask_sum

        return total / layer_count
