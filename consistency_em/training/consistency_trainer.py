"""Two-forward-pass HF Trainer subclass for consistency training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import Trainer

from consistency_em.training.loss import LossFn


class ConsistencyTrainer(Trainer):
    """HF ``Trainer`` that applies a pluggable paired-prompt consistency loss.

    Each batch carries four collated tensors —
    ``clean_input_ids``, ``clean_attention_mask``, ``wrapped_input_ids``
    and ``wrapped_attention_mask`` — emitted by
    :class:`consistency_em.data.paired_dataset.PairedDataCollator`. The
    configured :class:`LossFn` runs the two forward passes itself (the
    frozen clean target and the trainable wrapped pass) and returns the
    scalar loss that ``Trainer`` backprops through the wrapped pass.
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
        """Split the collated batch into clean / wrapped inputs and delegate to the loss.

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
        clean_inputs = {
            "input_ids": inputs["clean_input_ids"],
            "attention_mask": inputs["clean_attention_mask"],
        }
        wrapped_inputs = {
            "input_ids": inputs["wrapped_input_ids"],
            "attention_mask": inputs["wrapped_attention_mask"],
        }

        loss = self.loss_fn.compute(model, clean_inputs, wrapped_inputs)

        if return_outputs:
            return loss, None
        return loss
