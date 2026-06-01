"""Resolve a model-id string to its BaseModel singleton."""

from __future__ import annotations

from consistency_em.models.base_model import (
    GEMMA_2_9B,
    GPT_OSS_20B,
    LLAMA_3_1_8B,
    LLAMA_3_1_8B_INSTRUCT,
    LLAMA_3_2_1B,
    MISTRAL_7B_V0_3,
    BaseModel,
)

_BASE_MODELS = (
    GEMMA_2_9B,
    GPT_OSS_20B,
    LLAMA_3_1_8B,
    LLAMA_3_1_8B_INSTRUCT,
    LLAMA_3_2_1B,
    MISTRAL_7B_V0_3,
)
_BY_MODEL_ID = {model.model_id: model for model in _BASE_MODELS}


def base_model_for(model_id: str) -> BaseModel:
    """Return the BaseModel singleton with this ``model_id``.

    Raises:
        KeyError: If no registered model has the given id.
    """
    try:
        return _BY_MODEL_ID[model_id]
    except KeyError:
        raise KeyError(
            f"Unknown model id {model_id!r}; registered: {sorted(_BY_MODEL_ID)}"
        ) from None
