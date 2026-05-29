"""LossFn protocol — pluggable consistency-training objective."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


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
        clean_outputs: CausalLMOutputWithPast,
        wrapped_outputs: CausalLMOutputWithPast,
        clean_attention_mask: torch.Tensor,
        wrapped_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar loss tensor on the wrapped side's device.

        Args:
            clean_outputs: Forward-pass output from the clean pass.
                Activation losses read ``hidden_states``; logit losses
                read ``logits``.
            wrapped_outputs: Forward-pass output from the wrapped pass.
            clean_attention_mask: 2-D mask for the clean side.
            wrapped_attention_mask: 2-D mask for the wrapped side.

        Returns:
            Scalar tensor.
        """
        ...
