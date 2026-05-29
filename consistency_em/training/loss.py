"""LossFn protocol — pluggable consistency-training objective."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class LossFn(Protocol):
    """Compute a scalar consistency loss from paired clean/wrapped forward passes.

    A concrete loss runs the model on the clean inputs (a frozen
    target) and the wrapped inputs (trainable), reads whatever it needs
    from those passes — output logits, or hidden activations captured
    with forward hooks — and returns a single scalar tensor whose
    gradient is hooked into the wrapped side.
    """

    def compute(
        self,
        model: nn.Module,
        clean_inputs: dict[str, torch.Tensor],
        wrapped_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return a scalar loss tensor on the wrapped side's device.

        Args:
            model: The model under training.
            clean_inputs: ``input_ids`` and ``attention_mask`` for the
                clean (target) prompt.
            wrapped_inputs: ``input_ids`` and ``attention_mask`` for the
                wrapped (trained) prompt.

        Returns:
            Scalar tensor.
        """
        ...
