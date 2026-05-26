"""Tests for consistency_em._chat."""

from __future__ import annotations

from consistency_em._chat import render_messages
from tests.unit.conftest import _FakeTokenizer


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
