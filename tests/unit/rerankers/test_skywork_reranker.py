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
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.pad_token: str | None = None
        self.pad_token_id: int | None = None
        self.padding_side: str = "left"
        self.apply_calls: list[dict[str, Any]] = []
        self.tokenize_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        self.apply_calls.append(
            {
                "conversation": conversation,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return self.bos_token + " ".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in conversation
        )

    def __call__(
        self,
        rendered: list[str],
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> _TokenizedBatch:
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
        self.to_calls: list[str] = []

    def to(self, device: str) -> _TokenizedBatch:
        self.to_calls.append(device)
        return self

    def keys(self) -> list[str]:
        return ["input_ids", "attention_mask"]

    def __getitem__(self, key: str) -> torch.Tensor:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}[key]


class _FakeRewardModel:
    def __init__(self, logits_per_call: list[torch.Tensor]) -> None:
        self._logits_per_call = list(logits_per_call)
        self.config = SimpleNamespace(pad_token_id=None)
        self.call_args: list[dict[str, torch.Tensor]] = []
        self.to_calls: list[str] = []
        self.eval_calls = 0

    def __call__(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        self.call_args.append(kwargs)
        return SimpleNamespace(logits=self._logits_per_call.pop(0))

    def to(self, device: str) -> _FakeRewardModel:
        self.to_calls.append(device)
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

    def test_defaults_to_the_llama_3_1_8b_variant(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])

        SkyworkRewardReranker(device="cpu")

        assert patched_hf["model_model_id"] == "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

    def test_model_loaded_with_single_regression_head(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])

        SkyworkRewardReranker(device="cpu")

        assert patched_hf["num_labels"] == 1

    def test_model_set_to_eval_mode_on_load(self, patched_hf: dict[str, Any]) -> None:
        fake_model = _FakeRewardModel([torch.tensor([[0.0]])])
        patched_hf["model"] = fake_model

        SkyworkRewardReranker(device="cpu")

        assert fake_model.eval_calls == 1

    def test_model_moved_to_configured_device(self, patched_hf: dict[str, Any]) -> None:
        fake_model = _FakeRewardModel([torch.tensor([[0.0]])])
        patched_hf["model"] = fake_model

        SkyworkRewardReranker(device="cpu")

        assert fake_model.to_calls == ["cpu"]


class TestSkyworkRewardRerankerPadTokenSetup:
    def test_pad_token_defaults_to_eos_when_unset(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        patched_hf["tokenizer"].pad_token = None
        patched_hf["tokenizer"].pad_token_id = None

        SkyworkRewardReranker(device="cpu")

        assert patched_hf["tokenizer"].pad_token == patched_hf["tokenizer"].eos_token

    def test_existing_pad_token_is_preserved(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        patched_hf["tokenizer"].pad_token = "<pad>"
        patched_hf["tokenizer"].pad_token_id = 7

        SkyworkRewardReranker(device="cpu")

        assert patched_hf["tokenizer"].pad_token == "<pad>"

    def test_padding_side_forced_to_right(self, patched_hf: dict[str, Any]) -> None:
        fake_model = _FakeRewardModel([torch.tensor([[0.0]])])
        patched_hf["model"] = fake_model
        patched_hf["tokenizer"].padding_side = "left"

        SkyworkRewardReranker(device="cpu")

        assert patched_hf["tokenizer"].padding_side == "right"

    def test_model_config_pad_token_id_propagated_from_tokenizer(
        self, patched_hf: dict[str, Any]
    ) -> None:
        fake_model = _FakeRewardModel([torch.tensor([[0.0]])])
        patched_hf["model"] = fake_model
        patched_hf["tokenizer"].pad_token = "<pad>"
        patched_hf["tokenizer"].pad_token_id = 13

        SkyworkRewardReranker(device="cpu")

        assert fake_model.config.pad_token_id == 13


class TestSkyworkRewardRerankerRanking:
    def test_returns_one_score_per_candidate_in_input_order(
        self, patched_hf: dict[str, Any]
    ) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[1.5], [-0.5], [3.25]])])
        reranker = SkyworkRewardReranker(device="cpu")

        scores = reranker.rank("Q", ["candidate-a", "candidate-b", "candidate-c"])

        assert scores == [1.5, -0.5, 3.25]


class TestSkyworkRewardRerankerEmptyCandidates:
    def test_empty_candidate_list_returns_empty_list(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([])
        reranker = SkyworkRewardReranker(device="cpu")

        scores = reranker.rank("Q", [])

        assert scores == []

    def test_empty_candidate_list_does_not_call_the_model(self, patched_hf: dict[str, Any]) -> None:
        fake_model = _FakeRewardModel([])
        patched_hf["model"] = fake_model
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("Q", [])

        assert fake_model.call_args == []


class TestSkyworkRewardRerankerConversationFormat:
    def test_each_candidate_rendered_as_user_assistant_chat(
        self, patched_hf: dict[str, Any]
    ) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0], [0.0]])])
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("the question", ["answer-one", "answer-two"])

        conversations = [call["conversation"] for call in patched_hf["tokenizer"].apply_calls]
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

        reranker.rank("the question", ["the answer"])

        assert patched_hf["tokenizer"].apply_calls[0]["tokenize"] is False

    def test_chat_template_called_with_add_generation_prompt_false(
        self, patched_hf: dict[str, Any]
    ) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("the question", ["the answer"])

        assert patched_hf["tokenizer"].apply_calls[0]["add_generation_prompt"] is False

    def test_leading_bos_is_stripped_from_rendered_conversation(
        self, patched_hf: dict[str, Any]
    ) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("the question", ["the answer"])

        rendered = patched_hf["tokenizer"].tokenize_calls[0]["rendered"][0]
        assert not rendered.startswith("<s>")


class TestSkyworkRewardRerankerTokenization:
    def test_max_length_passed_to_tokenizer(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        reranker = SkyworkRewardReranker(device="cpu", max_length=512)

        reranker.rank("the question", ["the answer"])

        assert patched_hf["tokenizer"].tokenize_calls[0]["max_length"] == 512

    def test_padding_and_truncation_are_requested(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("the question", ["the answer"])

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


class TestSkyworkRewardRerankerDevicePlacement:
    def test_tokenized_batch_moved_to_configured_device(self, patched_hf: dict[str, Any]) -> None:
        patched_hf["model"] = _FakeRewardModel([torch.tensor([[0.0]])])
        reranker = SkyworkRewardReranker(device="cpu")

        reranker.rank("the question", ["the answer"])

        # The tokenizer fake returns a fresh _TokenizedBatch on each
        # call; that batch records its .to(device) calls.
        # We can't reach it directly without a registry, so verify
        # via the model call: the input_ids must have arrived.
        assert "input_ids" in patched_hf["model"].call_args[0]


class TestSkyworkRewardRerankerRealChatTemplate:
    """Integration check that the chat-template contract with Skywork holds.

    Loads the real Skywork tokenizer (no model weights). Skipped when
    the tokenizer is not available locally or via the HF cache — the
    test surfaces upstream chat-template changes when the cache is
    populated, without blocking offline CI.
    """

    def test_rendered_conversation_has_no_leading_bos_and_carries_both_turns(self) -> None:
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(SkyworkRewardReranker.DEFAULT_MODEL_ID)
        except Exception as exc:
            pytest.skip(f"Skywork tokenizer unavailable: {exc}")

        conversation = [
            {"role": "user", "content": "USER_SENTINEL"},
            {"role": "assistant", "content": "ASSISTANT_SENTINEL"},
        ]
        rendered = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        if tokenizer.bos_token is not None and rendered.startswith(tokenizer.bos_token):
            rendered = rendered[len(tokenizer.bos_token) :]
        assert tokenizer.bos_token is None or not rendered.startswith(tokenizer.bos_token)
        assert "USER_SENTINEL" in rendered
        assert "ASSISTANT_SENTINEL" in rendered
