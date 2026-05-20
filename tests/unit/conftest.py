"""Shared test fixtures and fakes for unit tests under ``tests/unit/``."""

from __future__ import annotations

import pytest


class _FakeTokenizer:
    """Minimal HF-tokenizer stub.

    Records every ``apply_chat_template`` call as a
    ``(messages, add_generation_prompt)`` tuple in
    ``apply_chat_template_calls`` for assertion. The rendered string
    is a recognizable marker built from the message contents.

    Also handles ``__call__`` (token-id encoding) by hashing each input
    string to a stable positive int; callers that pin specific token
    ids can override the per-string mapping via ``set_token_ids``.
    """

    def __init__(self, chat_template: str | None = "<template>") -> None:
        self.chat_template = chat_template
        self.apply_chat_template_calls: list[tuple[list[dict[str, str]], bool]] = []
        self._token_id_overrides: dict[str, list[int]] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.apply_chat_template_calls.append((messages, add_generation_prompt))
        return "<rendered: " + " | ".join(message["content"] for message in messages) + ">"

    def set_token_ids(self, text: str, token_ids: list[int]) -> None:
        self._token_id_overrides[text] = token_ids

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        if text in self._token_id_overrides:
            return {"input_ids": self._token_id_overrides[text]}
        # Deterministic positive int per string; sufficient for tests
        # that only care about ids being distinct and stable.
        return {"input_ids": [abs(hash(text)) % 100000]}


@pytest.fixture
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> _FakeTokenizer:
    """Monkeypatch ``AutoTokenizer.from_pretrained`` to return a fresh fake."""
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model_id: tokenizer,
    )
    return tokenizer
