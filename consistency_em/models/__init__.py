"""Model metadata — value objects for the HF models we use."""

from consistency_em.models.base_model import (
    GEMMA_2_9B,
    GPT_OSS_20B,
    LLAMA_3_1_8B,
    LLAMA_3_1_8B_INSTRUCT,
    LLAMA_3_2_1B,
    MISTRAL_7B_V0_3,
    BaseModel,
)
from consistency_em.models.lora_adapter import LoRAAdapter
from consistency_em.models.model_organism import ModelOrganism

__all__ = [
    "BaseModel",
    "GEMMA_2_9B",
    "GPT_OSS_20B",
    "LLAMA_3_1_8B",
    "LLAMA_3_1_8B_INSTRUCT",
    "LLAMA_3_2_1B",
    "LoRAAdapter",
    "MISTRAL_7B_V0_3",
    "ModelOrganism",
]
