"""Subject-model generators — produce completions for eval and labelling."""

from consistency_em.generation.vllm_generator import CompletionWithLogprob, VLLMGenerator

__all__ = ["CompletionWithLogprob", "VLLMGenerator"]
