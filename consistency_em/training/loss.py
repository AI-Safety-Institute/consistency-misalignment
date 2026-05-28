"""Consistency-training loss functions — pluggable ACT / BCT objectives."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class LossFn(Protocol):
    """Compute a scalar consistency loss from paired clean/wrapped forward outputs.

    A concrete loss consumes the two forward-pass outputs (clean is
    computed under ``torch.no_grad`` and treated as a frozen target;
    wrapped carries gradients) plus the attention masks needed to
    ignore padding, and returns a single scalar tensor with gradient
    hooked into the wrapped side.
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
                pass (a Hugging Face ``CausalLMOutput`` with ``logits``
                and / or ``hidden_states`` populated).
            wrapped_outputs: Forward-pass output object from the
                wrapped pass.
            clean_attention_mask: 2-D mask for the clean side.
            wrapped_attention_mask: 2-D mask for the wrapped side.

        Returns:
            Scalar tensor.
        """
        ...


def _connected_zero(reference: torch.Tensor) -> torch.Tensor:
    """Return a zero scalar that participates in ``reference``'s autograd graph.

    ``Trainer.training_step`` calls ``.backward()`` on whatever
    ``compute_loss`` returns. A bare ``torch.tensor(0.0)`` short-
    circuits there; multiplying by ``reference.sum()`` keeps the graph
    intact while still yielding a numerically-zero loss.
    """
    return 0.0 * reference.sum()


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
        clean_outputs: object,
        wrapped_outputs: object,
        clean_attention_mask: torch.Tensor,
        wrapped_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        clean_hidden = clean_outputs.hidden_states  # type: ignore[attr-defined]
        wrapped_hidden = wrapped_outputs.hidden_states  # type: ignore[attr-defined]

        if not clean_hidden or not wrapped_hidden:
            return torch.tensor(0.0, requires_grad=True)

        scale_factor = self.loss_scale**0.5
        layer_count = len(clean_hidden)
        total = _connected_zero(wrapped_hidden[0])

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


class BctLoss:
    """KL divergence with clean logits as soft labels, computed over the matched suffix.

    Standard distillation form: clean logits are detached (no grad)
    and softmaxed at temperature ``T`` to form soft targets; wrapped
    logits are log-softmaxed at the same temperature; the per-token
    cross-entropy is averaged over positions where both sides'
    attention masks are 1, then multiplied by ``T ** 2`` so the
    gradient magnitude stays comparable across temperatures
    (Hinton et al., 2015).
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

        if clean_logits.numel() == 0 or wrapped_logits.numel() == 0:
            return _connected_zero(wrapped_logits)

        suffix_len = min(clean_logits.size(1), wrapped_logits.size(1))
        clean_suffix_logits = clean_logits[:, -suffix_len:, :].detach()
        wrapped_suffix_logits = wrapped_logits[:, -suffix_len:, :]

        clean_probs = F.softmax(clean_suffix_logits / self.temperature, dim=-1)
        wrapped_log_probs = F.log_softmax(wrapped_suffix_logits / self.temperature, dim=-1)

        clean_suffix_mask = clean_attention_mask[:, -suffix_len:]
        wrapped_suffix_mask = wrapped_attention_mask[:, -suffix_len:]
        combined_mask = (clean_suffix_mask * wrapped_suffix_mask).float()

        if combined_mask.sum() == 0:
            return _connected_zero(wrapped_logits)

        per_token = -(clean_probs * wrapped_log_probs).sum(dim=-1)
        masked = per_token * combined_mask
        loss = masked.sum() / combined_mask.sum()
        return loss * (self.temperature**2)
