"""Tests for PairedDataCollator."""

from __future__ import annotations

import torch
from datasets import Dataset

from consistency_em.data.paired_dataset import PairedDataCollator


class _CharTokenizer:
    """Renders messages by joining contents, tokenizes one id per character."""

    chat_template = None

    def __call__(
        self, text: str, truncation: bool = False, max_length: int | None = None
    ) -> dict[str, list[int]]:
        token_ids = [ord(character) for character in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}


def _paired_row(clean_text: str, wrapped_text: str) -> dict:
    return {
        "clean_messages": [{"role": "user", "content": clean_text}],
        "wrapped_messages": [{"role": "user", "content": wrapped_text}],
    }


def _ids(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def _mask(length: int) -> torch.Tensor:
    return torch.ones(length, dtype=torch.long)


class TestPairedDataCollatorTokenize:
    def test_adds_the_four_token_columns_and_drops_messages(self) -> None:
        paired = Dataset.from_list([_paired_row("ab", "cd")])

        tokenized = PairedDataCollator.tokenize(paired, _CharTokenizer())

        assert sorted(tokenized.column_names) == [
            "clean_attention_mask",
            "clean_input_ids",
            "wrapped_attention_mask",
            "wrapped_input_ids",
        ]

    def test_tokenizes_clean_and_wrapped_sides_independently(self) -> None:
        paired = Dataset.from_list([_paired_row("ab", "xyz")])

        tokenized = PairedDataCollator.tokenize(paired, _CharTokenizer())

        assert tokenized[0]["clean_input_ids"] == [ord("a"), ord("b")]
        assert tokenized[0]["wrapped_input_ids"] == [ord("x"), ord("y"), ord("z")]

    def test_attention_mask_matches_input_ids_length(self) -> None:
        paired = Dataset.from_list([_paired_row("abc", "de")])

        tokenized = PairedDataCollator.tokenize(paired, _CharTokenizer())

        assert tokenized[0]["clean_attention_mask"] == [1, 1, 1]
        assert tokenized[0]["wrapped_attention_mask"] == [1, 1]

    def test_truncates_to_max_length(self) -> None:
        paired = Dataset.from_list([_paired_row("abcdef", "ghijkl")])

        tokenized = PairedDataCollator.tokenize(paired, _CharTokenizer(), max_length=3)

        assert tokenized[0]["clean_input_ids"] == [ord("a"), ord("b"), ord("c")]
        assert tokenized[0]["clean_attention_mask"] == [1, 1, 1]


class TestPairedDataCollator:
    def test_pads_clean_and_wrapped_separately(self) -> None:
        features = [
            {
                "clean_input_ids": _ids([1, 2]),
                "clean_attention_mask": _mask(2),
                "wrapped_input_ids": _ids([7, 8, 9]),
                "wrapped_attention_mask": _mask(3),
            },
            {
                "clean_input_ids": _ids([3, 4, 5, 6]),
                "clean_attention_mask": _mask(4),
                "wrapped_input_ids": _ids([10]),
                "wrapped_attention_mask": _mask(1),
            },
        ]

        batch = PairedDataCollator(pad_token_id=0)(features)

        assert batch["clean_input_ids"].shape == (2, 4)
        assert batch["wrapped_input_ids"].shape == (2, 3)
        assert batch["clean_input_ids"][0].tolist() == [1, 2, 0, 0]
        assert batch["wrapped_input_ids"][1].tolist() == [10, 0, 0]

    def test_pad_token_id_is_respected(self) -> None:
        features = [
            {
                "clean_input_ids": _ids([1]),
                "clean_attention_mask": _mask(1),
                "wrapped_input_ids": _ids([2, 3]),
                "wrapped_attention_mask": _mask(2),
            },
            {
                "clean_input_ids": _ids([4, 5, 6]),
                "clean_attention_mask": _mask(3),
                "wrapped_input_ids": _ids([7]),
                "wrapped_attention_mask": _mask(1),
            },
        ]

        batch = PairedDataCollator(pad_token_id=999)(features)

        assert batch["clean_input_ids"][0].tolist() == [1, 999, 999]
        assert batch["wrapped_input_ids"][1].tolist() == [7, 999]

    def test_attention_masks_pad_with_zeros(self) -> None:
        features = [
            {
                "clean_input_ids": _ids([1, 2]),
                "clean_attention_mask": _mask(2),
                "wrapped_input_ids": _ids([3]),
                "wrapped_attention_mask": _mask(1),
            },
            {
                "clean_input_ids": _ids([4]),
                "clean_attention_mask": _mask(1),
                "wrapped_input_ids": _ids([5, 6, 7]),
                "wrapped_attention_mask": _mask(3),
            },
        ]

        batch = PairedDataCollator(pad_token_id=0)(features)

        assert batch["clean_attention_mask"][1].tolist() == [1, 0]
        assert batch["wrapped_attention_mask"][0].tolist() == [1, 0, 0]
