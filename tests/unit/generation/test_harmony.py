"""Tests for the Harmony-output parser."""

from __future__ import annotations

from consistency_em.generation.harmony import extract_final_channel


class TestExtractFinalChannel:
    def test_returns_content_after_last_final_marker(self) -> None:
        text = "analysisLet me think...assistantfinalThe answer is 3"

        result = extract_final_channel(text)

        assert result == "The answer is 3"

    def test_returns_empty_when_truncated_before_final(self) -> None:
        # Real gpt-oss output truncated mid-analysis — no final channel reached.
        text = "analysisWe need to be careful here. The user is asking..."

        result = extract_final_channel(text)

        assert result == ""

    def test_passes_through_text_without_analysis_prefix(self) -> None:
        # Non-Harmony models (Llama, Gemma, Mistral) produce plain output.
        text = "The capital of France is Paris."

        result = extract_final_channel(text)

        assert result == "The capital of France is Paris."

    def test_strips_leading_whitespace_after_final_marker(self) -> None:
        text = "analysisthought...assistantfinal   The answer is 42"

        result = extract_final_channel(text)

        assert result == "The answer is 42"
