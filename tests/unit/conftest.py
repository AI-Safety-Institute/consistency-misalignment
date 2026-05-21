"""Shared test fixtures and fakes for unit tests."""

from __future__ import annotations

import pytest


class _FakeTokenizer:
    """Minimal HF-tokenizer stub used by VLLMGenerator tests.

    Records every apply_chat_template call for assertion, and returns
    deterministic token ids from __call__ — either a hash-derived id
    per text, or an override pinned via set_token_ids.

    Attributes:
        chat_template: Stand-in for the real tokenizer attribute.
        apply_chat_template_calls: List of (messages, add_generation_prompt)
            tuples recorded on each call.
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
        """Record the call and return a recognizable marker string.

        Args:
            messages: Chat-message list passed by the caller.
            tokenize: Whether the caller asked for token ids. Recorded
                via the call list, not consulted otherwise.
            add_generation_prompt: Whether the caller wants a trailing
                generation prompt.

        Returns:
            A marker string built from the message contents.
        """
        self.apply_chat_template_calls.append((messages, add_generation_prompt))
        return "<rendered: " + " | ".join(message["content"] for message in messages) + ">"

    def set_token_ids(self, text: str, token_ids: list[int]) -> None:
        """Pin the token-id sequence returned for one input string.

        Args:
            text: The exact input string to override.
            token_ids: The list of token ids to return when __call__
                is invoked with that text.
        """
        self._token_id_overrides[text] = token_ids

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        """Return token ids for one text input.

        Args:
            text: Input string to tokenize.
            add_special_tokens: Honored by the real tokenizer; ignored
                here because the fake never adds special tokens.

        Returns:
            A dict with an input_ids field. If set_token_ids has been
            called for this text, the pinned list is returned;
            otherwise a single hash-derived id is returned.
        """
        if text in self._token_id_overrides:
            return {"input_ids": self._token_id_overrides[text]}
        return {"input_ids": [abs(hash(text)) % 100000]}


@pytest.fixture
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> _FakeTokenizer:
    """Patch AutoTokenizer.from_pretrained to return a fresh fake tokenizer."""
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model_id: tokenizer,
    )
    return tokenizer
