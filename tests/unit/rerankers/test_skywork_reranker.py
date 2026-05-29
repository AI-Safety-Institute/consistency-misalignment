"""Unit tests for SkyworkRewardReranker — mocks HF model and tokenizer, no GPU."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from consistency_em.rerankers import Reranker
from consistency_em.rerankers import skywork_reranker as skywork_reranker_module
from consistency_em.rerankers.skywork_reranker import SkyworkRewardReranker


class _FakeTokenizer:
    def __init__(self) -> None:
        self.bos_token = "<s>"
        self.apply_calls: list[tuple[list[dict[str, str]], bool]] = []
        self.tokenize_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, conversation: list[dict[str, str]], tokenize: bool = True) -> str:
        self.apply_calls.append((conversation, tokenize))
        return self.bos_token + " ".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in conversation
        )

    def __call__(
        self,
        rendered: list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 4096,
    ) -> SimpleNamespace:
        self.tokenize_calls.append(
            {
                "rendered": rendered,
                "max_length": max_length,
                "padding": padding,
                "truncation": truncation,
            }
        )
        batch = len(rendered)
        return _TokenizedBatch(
            input_ids=torch.zeros(batch, 4, dtype=torch.long),
            attention_mask=torch.ones(batch, 4, dtype=torch.long),
        )


class _TokenizedBatch:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device: str) -> _TokenizedBatch:
        return self

    def keys(self) -> list[str]:
        return ["input_ids", "attention_mask"]

    def __getitem__(self, key: str) -> torch.Tensor:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}[key]


class _FakeRewardModel:
    def __init__(self, logits_per_call: list[torch.Tensor]) -> None:
        self._logits_per_call = list(logits_per_call)
        self.call_args: list[dict[str, torch.Tensor]] = []
        self.eval_calls = 0

    def __call__(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        self.call_args.append(kwargs)
        return SimpleNamespace(logits=self._logits_per_call.pop(0))

    def to(self, device: str) -> _FakeRewardModel:
        return self

    def eval(self) -> _FakeRewardModel:
        self.eval_calls += 1
        return self


@pytest.fixture
def patched_hf(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"tokenizer": _FakeTokenizer(), "model": None}

    def _build_tokenizer(model_id: str) -> _FakeTokenizer:
        state["tokenizer_model_id"] = model_id
        return state["tokenizer"]

    def _build_model(model_id: str, torch_dtype: torch.dtype, num_labels: int) -> _FakeRewardModel:
        state["model_model_id"] = model_id
        state["torch_dtype"] = torch_dtype
        state["num_labels"] = num_labels
        return state["model"]

    monkeypatch.setattr(
        skywork_reranker_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(_build_tokenizer),
    )
    monkeypatch.setattr(
        skywork_reranker_module.AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(_build_model),
    )
    return state


class TestSkyworkRewardRerankerProtocolConformance:
    def test_satisfies_the_reranker_protocol_structurally(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])

        reranker = SkyworkRewardReranker(device="cpu")

        assert isinstance(reranker, Reranker)


class TestSkyworkRewardRerankerLoading:
    def test_loads_the_configured_model_id(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])

        SkyworkRewardReranker(model_id="custom/reward-model", device="cpu")

        assert patched_hf["tokenizer_model_id"] == "custom/reward-model"
        assert patched_hf["model_model_id"] == "custom/reward-model"

    def test_defaults_to_the_qwen3_0_6b_variant(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])

        SkyworkRewardReranker(device="cpu")

        assert patched_hf["model_model_id"] == "Skywork/Skywork-Reward-V2-Qwen3-0.6B"

    def test_model_loaded_with_single_regression_head(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])

        SkyworkRewardReranker(device="cpu")

        assert patched_hf["num_labels"] == 1

    def test_model_set_to_eval_mode_on_load(self, patched_hf: dict[str, Any]) -> None:
        fake_model = _FakeRewardModel([torch.tensor([[0.0]])])
        patched_hf["model"] = fake_model

        SkyworkRewardReranker(device="cpu")

        assert fake_model.eval_calls == 1


class TestSkyworkRewardRerankerRanking:
    def test_returns_one_score_per_candidate_in_input_order(
        self, patched_hf: dict[str, Any]
    ) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[1.5], [-0.5], [3.25]])])
        reranker = SkyworkRewardReranker(device="cpu")

        scores = reranker.rank("Q", ["candidate-a", "candidate-b", "candidate-c"])

        assert scores == [1.5, -0.5, 3.25]

    def test_empty_candidate_list_returns_empty_without_calling_the_model(
        self, patched_hf: dict[str, Any]
    ) -> None:
        fake_model = _FakeRewardModel([])
        patched_hf["model"] = fake_model
        reranker = SkyworkRewardReranker(device="cpu")

        scores = reranker.rank("Q", [])

        assert scores == []
        assert fake_model.call_args == []


class TestSkyworkRewardRerankerConversationFormat:
    def test_each_candidate_rendered_as_user_assistant_chat(
        self, patched_hf: dict[str, Any]
    ) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0], [0.0]])])
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("the question", ["answer-one", "answer-two"])

        conversations = [call[0] for call in patched_hf["tokenizer"].apply_calls]
        assert conversations == [
            [
                {"role": "user", "content": "the question"},
                {"role": "assistant", "content": "answer-one"},
            ],
            [
                {"role": "user", "content": "the question"},
                {"role": "assistant", "content": "answer-two"},
            ],
        ]

    def test_chat_template_called_with_tokenize_false(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("Q", ["A"])

        assert patched_hf["tokenizer"].apply_calls[0][1] is False

    def test_leading_bos_is_stripped_from_rendered_conversation(
        self, patched_hf: dict[str, Any]
    ) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("Q", ["A"])

        rendered = patched_hf["tokenizer"].tokenize_calls[0]["rendered"][0]
        assert not rendered.startswith("<s>")


class TestSkyworkRewardRerankerTokenization:
    def test_max_length_passed_to_tokenizer(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        reranker = SkyworkRewardReranker(device="cpu", max_length=512)

        reranker.rank("Q", ["A"])

        assert patched_hf["tokenizer"].tokenize_calls[0]["max_length"] == 512

    def test_padding_and_truncation_are_requested(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("Q", ["A"])

        tokenize_call = patched_hf["tokenizer"].tokenize_calls[0]
        assert tokenize_call["padding"] is True
        assert tokenize_call["truncation"] is True


class TestSkyworkRewardRerankerBatching:
    def test_all_candidates_scored_in_a_single_forward_pass(
        self, patched_hf: dict[str, Any]
    ) -> None:
        fake_model = _FakeRewardModel([torch.tensor([[0.1], [0.2], [0.3], [0.4]])])
        patched_hf["model"] = fake_model
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("Q", ["a", "b", "c", "d"])

        assert len(fake_model.call_args) == 1
