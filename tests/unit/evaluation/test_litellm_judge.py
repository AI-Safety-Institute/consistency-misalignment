"""Unit tests for LiteLLMJudge."""

from __future__ import annotations

import types

import litellm
import pytest

from consistency_em.evaluation import JudgeResponse, LiteLLMJudge


def _fake_response(content: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


class _MockAcompletion:
    """Async mock for ``litellm.acompletion`` that records call kwargs."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> types.SimpleNamespace:
        self.calls.append(kwargs)
        return _fake_response(self.response_text)


class TestLiteLLMJudgeDefaults:
    def test_constructor_defaults(self) -> None:
        judge = LiteLLMJudge()

        assert judge.model == "openai/gpt-4o"
        assert judge.temperature == 0.0
        assert judge.max_tokens == 16
        assert judge.max_concurrent == 100


class TestLiteLLMJudgeRespondOne:
    def test_returns_parsed_score_when_text_contains_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _MockAcompletion(response_text="42")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        response = judge.respond_one("rubric", "", "")

        assert response == JudgeResponse(text="42", score=42.0)

    def test_returns_none_score_when_text_has_no_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _MockAcompletion(response_text="HARDCODED")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        response = judge.respond_one("rubric", "", "")

        assert response == JudgeResponse(text="HARDCODED", score=None)

    def test_parses_decimal_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _MockAcompletion(response_text="The score is 0.75")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        response = judge.respond_one("rubric", "", "")

        assert response.score == 0.75

    def test_strips_whitespace_around_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _MockAcompletion(response_text="  REFUSAL  \n")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        response = judge.respond_one("rubric", "", "")

        assert response.text == "REFUSAL"


class TestLiteLLMJudgeScoreOne:
    def test_returns_parsed_float_when_number_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _MockAcompletion(response_text="0.5")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        score = judge.score_one("rubric", "", "")

        assert score == 0.5

    def test_returns_zero_when_parse_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _MockAcompletion(response_text="AGREED")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        score = judge.score_one("rubric", "", "")

        assert score == 0.0


class TestLiteLLMJudgeScoreBatch:
    def test_preserves_input_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # max_concurrent=1 forces serial dispatch so the counter is
        # deterministic; asyncio.gather still returns results in
        # argument order, which is what we are verifying.
        counter = 0

        async def mock_acompletion(**kwargs):
            nonlocal counter
            counter += 1
            return _fake_response(str(counter))

        monkeypatch.setattr(litellm, "acompletion", mock_acompletion)
        judge = LiteLLMJudge(max_concurrent=1)
        prompts = ["", "", ""]
        completions = ["", "", ""]

        scores = judge.score_batch("rubric", prompts, completions)

        assert scores == [1.0, 2.0, 3.0]

    def test_calls_litellm_once_per_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _MockAcompletion(response_text="1.0")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()
        prompts = ["", "", "", "", ""]
        completions = ["", "", "", "", ""]

        results = judge.score_batch("rubric", prompts, completions)

        assert len(results) == 5
        assert len(mock.calls) == 5

    def test_length_mismatch_raises(self) -> None:
        judge = LiteLLMJudge()

        with pytest.raises(ValueError, match="len"):
            judge.score_batch("rubric", prompts=["one"], completions=["one", "two"])


class TestLiteLLMJudgeRespondBatch:
    def test_returns_one_judge_response_per_pair_in_input_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # max_concurrent=1 forces serial dispatch so the counter is
        # deterministic. The returned list must follow input order.
        counter = 0

        async def mock_acompletion(**kwargs):
            nonlocal counter
            counter += 1
            return _fake_response(f"score {counter}")

        monkeypatch.setattr(litellm, "acompletion", mock_acompletion)
        judge = LiteLLMJudge(max_concurrent=1)

        responses = judge.respond_batch("rubric", prompts=["", "", ""], completions=["", "", ""])

        assert responses == [
            JudgeResponse(text="score 1", score=1.0),
            JudgeResponse(text="score 2", score=2.0),
            JudgeResponse(text="score 3", score=3.0),
        ]

    def test_carries_raw_text_through_when_parse_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Multi-field rubric judges (like StrongREJECT) emit text
        # whose first number isn't the canonical score. respond_batch
        # must surface the raw text so the caller can run their own
        # parser.
        mock = _MockAcompletion(response_text="HARDCODED LABEL")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        responses = judge.respond_batch("rubric", prompts=[""], completions=[""])

        assert responses == [JudgeResponse(text="HARDCODED LABEL", score=None)]

    def test_empty_input_returns_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _MockAcompletion(response_text="never called")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        responses = judge.respond_batch("rubric", prompts=[], completions=[])

        assert responses == []
        assert mock.calls == []

    def test_length_mismatch_raises(self) -> None:
        judge = LiteLLMJudge()

        with pytest.raises(ValueError, match="len"):
            judge.respond_batch("rubric", prompts=["one"], completions=["one", "two"])


class TestLiteLLMJudgeRubricAuthoritative:
    def test_rubric_is_sent_as_user_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _MockAcompletion(response_text="1.0")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        judge.respond_one("THE_RUBRIC", "ignored_prompt", "ignored_completion")

        assert mock.calls[0]["messages"] == [{"role": "user", "content": "THE_RUBRIC"}]

    def test_falls_back_to_prompt_plus_completion_when_rubric_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _MockAcompletion(response_text="1.0")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge()

        judge.respond_one("", "the prompt", "the completion")

        content = mock.calls[0]["messages"][0]["content"]
        assert "the prompt" in content
        assert "the completion" in content

    def test_passes_model_temperature_and_max_tokens_to_litellm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _MockAcompletion(response_text="1.0")
        monkeypatch.setattr(litellm, "acompletion", mock)
        judge = LiteLLMJudge(model="anthropic/claude-3-5-sonnet", temperature=0.2, max_tokens=8)

        judge.respond_one("rubric", "", "")

        assert mock.calls[0]["model"] == "anthropic/claude-3-5-sonnet"
        assert mock.calls[0]["temperature"] == 0.2
        assert mock.calls[0]["max_tokens"] == 8


@pytest.mark.slow
class TestLiteLLMJudgeRealApi:
    def test_real_call_smoke(self) -> None:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set; skipping real-API smoke test")

        judge = LiteLLMJudge(model="openai/gpt-4o-mini")

        response = judge.respond_one(
            rubric="Respond with the single integer 42 and nothing else.",
            prompt="",
            completion="",
        )

        assert response.score == 42.0
