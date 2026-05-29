"""Two-forward-pass HF Trainer subclass for consistency training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import Trainer

from consistency_em.training.loss import LossFn


class ConsistencyTrainer(Trainer):
    """HF ``Trainer`` that does two forward passes per batch and applies a pluggable loss.

    Each batch carries four collated tensors —
    ``clean_input_ids``, ``clean_attention_mask``, ``wrapped_input_ids``
    and ``wrapped_attention_mask`` — emitted by
    :class:`consistency_em.data.paired_dataset.PairedDataCollator`. The
    clean side is run in eval mode under ``torch.no_grad`` to produce a
    deterministic frozen target (dropout off); the wrapped side runs in
    train mode with gradients. The configured :class:`LossFn` consumes
    both outputs and returns the scalar loss that ``Trainer`` then
    backprops through the wrapped pass.

    ``output_hidden_states=True`` is requested on both forward passes
    so activation-style losses can read ``outputs.hidden_states``;
    logit-style losses ignore that field and read ``outputs.logits``.
    """

    def __init__(self, loss_fn: LossFn, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        """Run the clean (frozen) and wrapped forward passes and apply the configured loss.

        Args:
            model: The model under training.
            inputs: A collated batch carrying ``clean_input_ids``,
                ``clean_attention_mask``, ``wrapped_input_ids`` and
                ``wrapped_attention_mask``.
            return_outputs: When True, return ``(loss, None)`` to match
                the ``Trainer`` contract (there is no single outputs
                object to surface across the two passes).
            num_items_in_batch: Accepted for ``Trainer`` compatibility;
                unused because the loss already reduces over the batch.

        Returns:
            The scalar loss, or ``(loss, None)`` when ``return_outputs``.
        """
        clean_input_ids = inputs["clean_input_ids"]
        clean_attention_mask = inputs["clean_attention_mask"]
        wrapped_input_ids = inputs["wrapped_input_ids"]
        wrapped_attention_mask = inputs["wrapped_attention_mask"]

        # The clean side is a frozen target (soft labels for BCT, reference
        # activations for ACT), so run it in eval mode to disable dropout —
        # including LoRA dropout — and keep the target deterministic. Restore
        # train mode for the gradient-bearing wrapped pass.
        model.eval()
        with torch.no_grad():
            clean_outputs = model(
                input_ids=clean_input_ids,
                attention_mask=clean_attention_mask,
                output_hidden_states=True,
            )

        model.train()
        wrapped_outputs = model(
            input_ids=wrapped_input_ids,
            attention_mask=wrapped_attention_mask,
            output_hidden_states=True,
        )

        loss = self.loss_fn.compute(
            clean_outputs=clean_outputs,
            wrapped_outputs=wrapped_outputs,
            clean_attention_mask=clean_attention_mask,
            wrapped_attention_mask=wrapped_attention_mask,
        )

        if return_outputs:
            return loss, None
        return loss
