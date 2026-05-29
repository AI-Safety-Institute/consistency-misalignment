"""BctLoss — KL divergence with clean logits as soft labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from consistency_em.training.loss import connected_zero


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
            return connected_zero(wrapped_logits)

        suffix_len = min(clean_logits.size(1), wrapped_logits.size(1))
        clean_suffix_logits = clean_logits[:, -suffix_len:, :].detach()
        wrapped_suffix_logits = wrapped_logits[:, -suffix_len:, :]

        clean_probs = F.softmax(clean_suffix_logits / self.temperature, dim=-1)
        wrapped_log_probs = F.log_softmax(wrapped_suffix_logits / self.temperature, dim=-1)

        clean_suffix_mask = clean_attention_mask[:, -suffix_len:]
        wrapped_suffix_mask = wrapped_attention_mask[:, -suffix_len:]
        combined_mask = (clean_suffix_mask * wrapped_suffix_mask).float()

        if combined_mask.sum() == 0:
            return connected_zero(wrapped_logits)

        per_token = -(clean_probs * wrapped_log_probs).sum(dim=-1)
        masked = per_token * combined_mask
        loss = masked.sum() / combined_mask.sum()
        return loss * (self.temperature**2)
