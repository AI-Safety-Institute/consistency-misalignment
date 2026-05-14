"""Unit tests for BaseModel and its concrete singletons."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from consistency_em.models import (
    GEMMA_2_9B,
    GPT_OSS_20B,
    LLAMA_3_1_8B,
    LLAMA_3_1_8B_INSTRUCT,
    LLAMA_3_2_1B,
    MISTRAL_7B_V0_3,
    BaseModel,
)


class TestBaseModelDataclass:
    def test_default_field_values(self) -> None:
        model = BaseModel(model_id="org/some-model")

        assert model.enforce_eager is False
        assert model.attention_backend == "default"

    def test_is_frozen(self) -> None:
        model = BaseModel(model_id="org/some-model")

        with pytest.raises(FrozenInstanceError):
            model.model_id = "other/model"  # type: ignore[misc]


class TestConcreteSingletons:
    def test_llama_3_2_1b(self) -> None:
        assert LLAMA_3_2_1B.model_id == "meta-llama/Llama-3.2-1B"
        assert LLAMA_3_2_1B.enforce_eager is False
        assert LLAMA_3_2_1B.attention_backend == "default"

    def test_llama_3_1_8b(self) -> None:
        assert LLAMA_3_1_8B.model_id == "meta-llama/Llama-3.1-8B"
        assert LLAMA_3_1_8B.enforce_eager is False
        assert LLAMA_3_1_8B.attention_backend == "default"

    def test_llama_3_1_8b_instruct(self) -> None:
        assert LLAMA_3_1_8B_INSTRUCT.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert LLAMA_3_1_8B_INSTRUCT.enforce_eager is False
        assert LLAMA_3_1_8B_INSTRUCT.attention_backend == "default"

    def test_gemma_2_9b_has_special_flags(self) -> None:
        assert GEMMA_2_9B.model_id == "google/gemma-2-9b"
        assert GEMMA_2_9B.enforce_eager is True
        assert GEMMA_2_9B.attention_backend == "FLASHINFER"

    def test_gpt_oss_20b(self) -> None:
        assert GPT_OSS_20B.model_id == "openai/gpt-oss-20b"
        assert GPT_OSS_20B.enforce_eager is False
        assert GPT_OSS_20B.attention_backend == "default"

    def test_mistral_7b_v0_3(self) -> None:
        assert MISTRAL_7B_V0_3.model_id == "mistralai/Mistral-7B-v0.3"
        assert MISTRAL_7B_V0_3.enforce_eager is False
        assert MISTRAL_7B_V0_3.attention_backend == "default"
