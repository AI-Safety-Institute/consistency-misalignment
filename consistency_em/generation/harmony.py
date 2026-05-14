"""Output-format parsing for OpenAI Harmony chat responses.

Models that use the Harmony chat format (e.g. ``openai/gpt-oss-*``)
emit responses across multiple channels: ``analysis`` for chain-of-
thought reasoning, ``commentary`` for tool-use scaffolding, and
``final`` for the user-facing answer. vLLM decodes the channel
boundary tokens to plain text, so a typical output looks like
``"analysisreasoning words ... assistantfinalthe answer"``.

This module surfaces just the user-facing answer for scoring layers
that expect a clean completion string regardless of which model
produced it.
"""

from __future__ import annotations


def extract_final_channel(text: str) -> str:
    """Return the user-facing ``final`` channel from Harmony output.

    If ``text`` starts with ``analysis``, returns everything after
    the last ``final`` word — the user-facing answer in Harmony's
    channel scheme. When the response is truncated mid-analysis
    with no ``final`` reached, returns the empty string so scoring
    layers see a missing answer rather than reasoning prose.

    For models without channel markers (Llama, Gemma, Mistral), the
    text is returned unchanged.
    """
    if not text.startswith("analysis"):
        return text
    final_marker = text.rfind("final")
    if final_marker == -1:
        return ""
    return text[final_marker + len("final") :].lstrip()
