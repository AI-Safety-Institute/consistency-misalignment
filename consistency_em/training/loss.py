"""Consistency-training loss functions — pluggable ACT / BCT objectives."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class LossFn(Protocol):
    """Compute a scalar consistency loss from paired clean/wrapped forward outputs.

    A concrete loss consumes the two forward-pass outputs (clean is
    typically computed under ``torch.no_grad`` and treated as a frozen
    target; wrapped carries gradients) plus the attention masks needed
    to ignore padding, and returns a single scalar tensor with
    gradient hooked into the wrapped side.
    """

    def compute(
        self,
        clean_outputs: object,
        wrapped_outputs: object,
        clean_attention_mask: torch.Tensor,
        wrapped_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar loss tensor on the wrapped side's device.

        Args:
            clean_outputs: Forward-pass output object from the clean
                pass (typically a Hugging Face ``CausalLMOutput`` with
                ``logits`` and / or ``hidden_states`` populated).
            wrapped_outputs: Forward-pass output object from the
                wrapped pass.
            clean_attention_mask: 2-D mask for the clean side.
            wrapped_attention_mask: 2-D mask for the wrapped side.

        Returns:
            Scalar tensor.
        """
        ...


class ActLoss:
    """L2 distance between matched-suffix hidden activations across all layers.

    Both forwards must run with ``output_hidden_states=True``. The two
    sides typically have different sequence lengths (the wrapped prompt
    carries an extra adversarial preamble or similar), so each layer's
    pair is aligned on the shorter side's length — the trailing
    ``suffix_len`` positions of each.

    The squared difference is scaled by ``scale_factor = sqrt(loss_scale)``
    **before** squaring to keep the running sum finite when raw
    activations have large magnitudes; the net effect is
    ``loss_scale * (diff ** 2).mean()`` per layer, averaged across
    layers.
    """

    def __init__(self, loss_scale: float = 1e-4) -> None:
        self.loss_scale = loss_scale

    def compute(
        self,
        clean_outputs: object,
        wrapped_outputs: object,
        clean_attention_mask: torch.Tensor,
        wrapped_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        clean_hidden = clean_outputs.hidden_states  # type: ignore[attr-defined]
        wrapped_hidden = wrapped_outputs.hidden_states  # type: ignore[attr-defined]

        if not clean_hidden or not wrapped_hidden:
            return torch.tensor(0.0, requires_grad=True)

        device = wrapped_hidden[0].device
        scale_factor = self.loss_scale**0.5
        total = torch.tensor(0.0, device=device)
        layer_count = 0
        for clean_layer, wrapped_layer in zip(clean_hidden, wrapped_hidden, strict=True):
            suffix_len = min(clean_layer.size(1), wrapped_layer.size(1))
            clean_suffix = clean_layer[:, -suffix_len:, :].detach().to(device)
            wrapped_suffix = wrapped_layer[:, -suffix_len:, :]
            diff = wrapped_suffix - clean_suffix
            total = total + (scale_factor * diff).pow(2).mean()
            layer_count += 1

        return total / layer_count


class BctLoss:
    """KL divergence with clean logits as soft labels, computed over the matched suffix.

    Standard distillation form: clean logits are detached (no grad)
    and softmaxed at temperature ``T`` to form soft targets; wrapped
    logits are log-softmaxed at the same temperature; the per-token
    cross-entropy ``-sum(p_clean * log p_wrapped)`` is averaged over
    positions where both sides' attention masks are 1, then multiplied
    by ``T ** 2`` to match the original gradient magnitude.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def compute(
        self,
        clean_outputs: object,
        wrapped_outputs: object,
        clean_attention_mask: torch.Tensor,
        wrapped_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        clean_logits = clean_outputs.logits  # type: ignore[attr-defined]
        wrapped_logits = wrapped_outputs.logits  # type: ignore[attr-defined]
        device = wrapped_logits.device

        if clean_logits.numel() == 0 or wrapped_logits.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        suffix_len = min(clean_logits.size(1), wrapped_logits.size(1))
        clean_suffix_logits = clean_logits[:, -suffix_len:, :].detach()
        wrapped_suffix_logits = wrapped_logits[:, -suffix_len:, :]

        clean_probs = F.softmax(clean_suffix_logits / self.temperature, dim=-1)
        wrapped_log_probs = F.log_softmax(wrapped_suffix_logits / self.temperature, dim=-1)

        clean_suffix_mask = clean_attention_mask[:, -suffix_len:]
        wrapped_suffix_mask = wrapped_attention_mask[:, -suffix_len:]
        combined_mask = (clean_suffix_mask * wrapped_suffix_mask).float()

        if combined_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        per_token = -(clean_probs * wrapped_log_probs).sum(dim=-1)
        masked = per_token * combined_mask
        loss = masked.sum() / combined_mask.sum()
        return loss * (self.temperature**2)
