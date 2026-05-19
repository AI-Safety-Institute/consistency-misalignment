"""Shared test fixtures and fakes for unit tests under ``tests/unit/``."""

from __future__ import annotations

import pytest


class _FakeTokenizer:
    """Minimal HF-tokenizer stub.

    Records every ``apply_chat_template`` call as a
    ``(messages, add_generation_prompt)`` tuple in
    ``apply_chat_template_calls`` for assertion. The rendered string
    is a recognizable marker built from the message contents.
    """

    def __init__(self, chat_template: str | None = "<template>") -> None:
        self.chat_template = chat_template
        self.apply_chat_template_calls: list[tuple[list[dict[str, str]], bool]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.apply_chat_template_calls.append((messages, add_generation_prompt))
        return "<rendered: " + " | ".join(message["content"] for message in messages) + ">"


@pytest.fixture
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> _FakeTokenizer:
    """Monkeypatch ``AutoTokenizer.from_pretrained`` to return a fresh fake."""
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model_id: tokenizer,
    )
    return tokenizer
