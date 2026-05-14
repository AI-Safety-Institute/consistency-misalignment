"""Tests for PairedDataCollator."""

from __future__ import annotations

import torch

from consistency_em.data.paired_dataset import PairedDataCollator


def _ids(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def _mask(length: int) -> torch.Tensor:
    return torch.ones(length, dtype=torch.long)


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
