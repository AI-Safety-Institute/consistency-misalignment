"""Paired forward-pass helpers shared by the consistency losses."""

from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


def clean_pass(
    model: nn.Module, inputs: dict[str, torch.Tensor], **forward_kwargs: object
) -> CausalLMOutputWithPast:
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
) -> CausalLMOutputWithPast:
    """Run the trainable forward: train mode (dropout on), gradients enabled."""
    model.train()
    return model(**inputs, **forward_kwargs)
