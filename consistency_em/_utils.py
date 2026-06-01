"""Cross-cutting helpers shared across data, generation, evaluation, and training."""

from __future__ import annotations

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase


def mean_or_zero(values: Sequence[float]) -> float:
    """Mean of ``values``, or ``0.0`` when ``values`` is empty.

    Returns ``0.0`` for an empty category instead of raising
    ``ZeroDivisionError``.
    """
    return sum(values) / len(values) if values else 0.0


def chunked(sequence: list, chunk_size: int) -> list[list]:
    """Reshape a flat list into consecutive ``chunk_size``-element chunks."""
    return [sequence[start : start + chunk_size] for start in range(0, len(sequence), chunk_size)]


def render_messages(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool,
) -> str:
    """Render a chat-message list into the string the model expects.

    Applies the tokenizer's chat template when present (Instruct
    models); falls back to a plain ``"\\n\\n".join`` of the message
    contents otherwise (base models that ship without a template).

    Messages missing the ``role`` key are treated as ``"user"`` so
    chat templates that access ``message.role`` don't crash on rows
    shipped without role labels.

    Args:
        messages: Chat-message list, e.g.
            ``[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]``.
        tokenizer: The HF tokenizer for the model the string is destined for.
        add_generation_prompt: ``True`` to append the assistant-turn
            opener so the model continues from the last user turn.
            ``False`` to render the messages as-is.

    Returns:
        A single rendered string ready to be tokenized.
    """
    normalized = [
        message if "role" in message else {"role": "user", **message} for message in messages
    ]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    return "\n\n".join(message["content"] for message in normalized)


def prompt_only_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Strip non-user turns from a chat-format messages list.

    The shipped misalignment datasets carry ``[user, assistant]`` rows
    where the assistant turn is reference content, not a target. When the
    model is asked to generate a fresh response, that prior assistant
    content would otherwise condition the generation as a continuation
    rather than a new answer. Returns the ``role == "user"`` turns.

    Some shipped eval sets omit role keys entirely
    (``[{content: prompt}, {content: reference}]``); for those the first
    message is the prompt by convention and is returned as the lone user
    turn, mirroring how ``render_messages`` treats role-less messages.

    Raises:
        ValueError: If roles are present but none is ``user``.
    """
    user_turns = [message for message in messages if message.get("role") == "user"]
    if user_turns:
        return user_turns
    if messages and all("role" not in message for message in messages):
        return [{"role": "user", **messages[0]}]
    raise ValueError(f"messages contain no role='user' turn: {messages!r}")
