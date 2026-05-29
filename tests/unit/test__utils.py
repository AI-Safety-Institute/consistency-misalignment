"""Tests for consistency_em._utils."""

from __future__ import annotations

import pytest

from consistency_em._utils import chunked, mean_or_zero, prompt_only_messages, render_messages
from tests.unit.conftest import _FakeTokenizer


class TestChunked:
    def test_reshapes_into_consecutive_chunks(self) -> None:
        reshaped = chunked([1, 2, 3, 4, 5, 6], 2)

        assert reshaped == [[1, 2], [3, 4], [5, 6]]

    def test_chunk_size_equal_to_length_returns_one_chunk(self) -> None:
        reshaped = chunked([1, 2, 3], 3)

        assert reshaped == [[1, 2, 3]]

    def test_empty_sequence_returns_empty_list(self) -> None:
        reshaped = chunked([], 4)

        assert reshaped == []

    def test_ragged_tail_kept_as_short_chunk(self) -> None:
        reshaped = chunked([1, 2, 3, 4, 5], 2)

        assert reshaped == [[1, 2], [3, 4], [5]]


class TestMeanOrZero:
    def test_empty_returns_zero(self) -> None:
        assert mean_or_zero([]) == 0.0

    def test_floats(self) -> None:
        assert mean_or_zero([0.0, 0.5, 1.0]) == 0.5

    def test_bools(self) -> None:
        # Booleans sum as ints; mean is the fraction True.
        assert mean_or_zero([True, False, True, True]) == 0.75

    def test_single_value(self) -> None:
        assert mean_or_zero([0.7]) == 0.7


class TestRenderMessages:
    def test_applies_chat_template_when_tokenizer_has_one(self) -> None:
        tokenizer = _FakeTokenizer()

        rendered = render_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            tokenizer,
            add_generation_prompt=True,
        )

        assert rendered == "<rendered: hello | hi there>"
        assert tokenizer.apply_chat_template_calls[-1][1] is True

    def test_passes_add_generation_prompt_false_through(self) -> None:
        tokenizer = _FakeTokenizer()

        render_messages(
            [{"role": "user", "content": "hello"}],
            tokenizer,
            add_generation_prompt=False,
        )

        assert tokenizer.apply_chat_template_calls[-1][1] is False

    def test_falls_back_to_plain_join_when_tokenizer_has_no_chat_template(self) -> None:
        tokenizer = _FakeTokenizer(chat_template=None)

        rendered = render_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            tokenizer,
            add_generation_prompt=True,
        )

        assert rendered == "hello\n\nhi there"
        assert tokenizer.apply_chat_template_calls == []

    def test_defaults_missing_role_to_user(self) -> None:
        tokenizer = _FakeTokenizer()

        render_messages(
            [
                {"content": "no role here"},
                {"role": "assistant", "content": "ok"},
            ],
            tokenizer,
            add_generation_prompt=False,
        )

        sent_messages, _ = tokenizer.apply_chat_template_calls[-1]
        assert sent_messages == [
            {"role": "user", "content": "no role here"},
            {"role": "assistant", "content": "ok"},
        ]


class TestPromptOnlyMessages:
    def test_assistant_turn_after_user_is_dropped(self) -> None:
        # The shape every shipped consistency.jsonl / act_bct_*.jsonl row
        # carries: user question + reference assistant turn. The labeller
        # must not condition the model on the prior assistant content.
        messages = [
            {"role": "user", "content": "the question"},
            {"role": "assistant", "content": "POISONED prior response"},
        ]

        sliced = prompt_only_messages(messages)

        assert sliced == [{"role": "user", "content": "the question"}]

    def test_no_user_turn_raises_value_error(self) -> None:
        # Defensive: a malformed row with no user turn would silently
        # produce empty prompts otherwise.
        messages = [{"role": "assistant", "content": "no user here"}]

        with pytest.raises(ValueError, match="no role='user' turn"):
            prompt_only_messages(messages)
