"""Subject-model metadata as a value object.

``BaseModel`` carries the small set of facts a generator or trainer
needs about an HF model — its repo id and a couple of vLLM-loading
quirks specific to certain architectures. Construction is cheap and
has no side effects, so the same singleton can be reused across
generators, trainers, and configs.

Field set is intentionally minimal. Additional fields (LoRA target
modules, FSDP block class, default LoRA rank, parameter count) will
join the dataclass when the consumer that needs them lands — see
``todo.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseModel:
    """HF model identity plus the few quirks vLLM needs at load time.

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
