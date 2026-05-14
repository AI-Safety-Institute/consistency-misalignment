"""Model metadata as a value object.

``BaseModel`` carries the small set of facts a generator or trainer
needs about an HF model — its repo id and a couple of vLLM-loading
flags specific to certain architectures. The same instance is reused
across the lifecycle: pre-training to evaluate the base weights, mid-
training to evaluate an in-progress LoRA adapter against the same
base, and post-training to evaluate the final organism.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseModel:
    """HF model identity plus the few flags vLLM needs at load time.

    Attributes:
        model_id: The HF repo id passed to ``vllm.LLM(model=...)``
            and to the HF ``AutoTokenizer``. Also used as the on-disk
            cache key.
        enforce_eager: Disables vLLM's CUDA-graph compilation. Some
            architectures (notably Gemma-2 with its logit
            soft-capping) aren't supported by Flash Attention CUDA
            graphs and require this. Leave ``False`` for everything
            else; eager mode is slower.
        attention_backend: The vLLM attention backend. ``"default"``
            lets vLLM pick. Gemma-2 needs ``"FLASHINFER"`` because
            FlashInfer is the backend that implements the tanh
            soft-capping logic Gemma-2's attention requires.
    """

    model_id: str
    enforce_eager: bool = False
    attention_backend: str = "default"


LLAMA_3_2_1B = BaseModel(model_id="meta-llama/Llama-3.2-1B")
LLAMA_3_1_8B = BaseModel(model_id="meta-llama/Llama-3.1-8B")
LLAMA_3_1_8B_INSTRUCT = BaseModel(model_id="meta-llama/Llama-3.1-8B-Instruct")
GEMMA_2_9B = BaseModel(
    model_id="google/gemma-2-9b",
    enforce_eager=True,
    attention_backend="FLASHINFER",
)
GPT_OSS_20B = BaseModel(model_id="openai/gpt-oss-20b")
MISTRAL_7B_V0_3 = BaseModel(model_id="mistralai/Mistral-7B-v0.3")
