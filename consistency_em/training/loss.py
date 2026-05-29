"""LossFn protocol — pluggable consistency-training objective."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class LossFn(Protocol):
    """Compute a scalar consistency loss from paired clean/wrapped forward passes.

    A concrete loss runs the model on the clean inputs (a frozen
    target, via :func:`clean_pass`) and the wrapped inputs (trainable,
    via :func:`wrapped_pass`), reads whatever it needs from those
    passes — output logits, or hidden activations captured with forward
    hooks — and returns a single scalar tensor whose gradient is hooked
    into the wrapped side.
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


def clean_pass(
    model: nn.Module, inputs: dict[str, torch.Tensor], **forward_kwargs: object
) -> object:
    """Run the frozen-target forward: eval mode (dropout off), no gradients.

    The clean side is a target the wrapped side is trained to match, so
    dropout — including LoRA dropout — must be off to keep it
    deterministic.
    """
    model.eval()
    with torch.no_grad():
        return model(**inputs, **forward_kwargs)


def wrapped_pass(
    model: nn.Module, inputs: dict[str, torch.Tensor], **forward_kwargs: object
) -> object:
    """Run the trainable forward: train mode (dropout on), gradients enabled."""
    model.train()
    return model(**inputs, **forward_kwargs)
