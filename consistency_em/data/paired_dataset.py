"""Paired-batch collator for ACT/BCT consistency training.

Sits downstream of :attr:`MisalignmentDataset.act_bct_dataset` (where
the per-task clean / wrapped prompt construction happens) and a
tokenization step that adds the four token columns
(``clean_input_ids``, ``clean_attention_mask``, ``wrapped_input_ids``,
``wrapped_attention_mask``) to each row.

The collator pads clean and wrapped sequences separately within a
batch, because the two sides typically have different lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset

from consistency_em._utils import render_messages


def tokenize_paired_dataset(
    paired_dataset: Dataset,
    tokenizer: object,
    max_length: int = 1024,
) -> Dataset:
    """Add the four token columns the paired collator consumes.

    Renders each row's ``clean_messages`` and ``wrapped_messages``
    through the chat template and tokenizes them (truncating to
    ``max_length``), yielding ``clean_input_ids`` / ``clean_attention_mask``
    / ``wrapped_input_ids`` / ``wrapped_attention_mask`` and dropping the
    original columns. The assistant turn is already present in both
    sides, so no generation prompt is appended.
    """

    def tokenize_row(row: dict) -> dict:
        clean = tokenizer(
            render_messages(row["clean_messages"], tokenizer, add_generation_prompt=False),
            truncation=True,
            max_length=max_length,
        )
        wrapped = tokenizer(
            render_messages(row["wrapped_messages"], tokenizer, add_generation_prompt=False),
            truncation=True,
            max_length=max_length,
        )
        return {
            "clean_input_ids": clean["input_ids"],
            "clean_attention_mask": clean["attention_mask"],
            "wrapped_input_ids": wrapped["input_ids"],
            "wrapped_attention_mask": wrapped["attention_mask"],
        }

    return paired_dataset.map(tokenize_row, remove_columns=paired_dataset.column_names)


@dataclass
class PairedDataCollator:
    """Pad clean and wrapped sequences separately for a paired-batch forward.

    The trainer needs both the clean and the wrapped sequence in each batch,
    but they may have different lengths. This collator pads each side to its
    own max length within the batch.

    Attributes:
        pad_token_id: Token id used for padding. No default — must be set
            per tokenizer (typically the tokenizer's ``pad_token_id``, or
            the ``eos_token_id`` for models without a dedicated pad token).
    """

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate paired tokenized examples into a padded batch.

        Args:
            features: A list of dicts, each with ``clean_input_ids``,
                ``clean_attention_mask``, ``wrapped_input_ids``,
                ``wrapped_attention_mask`` keys mapping to 1-D tensors or
                tensor-like sequences of token ids / mask values.

        Returns:
            A dict mapping each of the four paired keys to a 2-D
            ``torch.Tensor`` of shape ``(batch, max_len_for_that_side)``,
            with dtype ``int64``.
        """
        clean_ids = [feature["clean_input_ids"] for feature in features]
        clean_masks = [feature["clean_attention_mask"] for feature in features]
        wrapped_ids = [feature["wrapped_input_ids"] for feature in features]
        wrapped_masks = [feature["wrapped_attention_mask"] for feature in features]

        padded_clean_ids, padded_clean_masks = self._pad_side(
            clean_ids, clean_masks, self.pad_token_id
        )
        padded_wrapped_ids, padded_wrapped_masks = self._pad_side(
            wrapped_ids, wrapped_masks, self.pad_token_id
        )

        return {
            "clean_input_ids": torch.stack(padded_clean_ids).long(),
            "clean_attention_mask": torch.stack(padded_clean_masks).long(),
            "wrapped_input_ids": torch.stack(padded_wrapped_ids).long(),
            "wrapped_attention_mask": torch.stack(padded_wrapped_masks).long(),
        }

    @staticmethod
    def _pad_side(
        ids: list[torch.Tensor],
        masks: list[torch.Tensor],
        pad_token_id: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Right-pad one side (clean or wrapped) of the batch to a common length.

        All sequences in ``ids`` are padded to the longest sequence in the
        batch using ``pad_token_id``; the corresponding ``masks`` are padded
        with zeros (so attention ignores the padding positions). ``ids`` and
        ``masks`` must be the same outer length and paired by index.

        Args:
            ids: Per-row token-id sequences, possibly of different lengths.
            masks: Per-row attention masks, paired with ``ids`` by index.
            pad_token_id: Value used to fill padded positions in ``ids``;
                attention masks are always padded with zeros regardless.

        Returns:
            A ``(padded_ids, padded_masks)`` pair of lists. Both lists have
            the same length as their inputs, and every tensor in either list
            is now of length ``max(len(seq) for seq in ids)``.
        """
        max_len = max(len(seq) for seq in ids)
        padded_ids: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        for seq_ids, seq_mask in zip(ids, masks, strict=True):
            pad_len = max_len - len(seq_ids)
            if pad_len > 0:
                padded_ids.append(
                    torch.cat(
                        [
                            torch.as_tensor(seq_ids),
                            torch.full((pad_len,), pad_token_id),
                        ]
                    )
                )
                padded_masks.append(
                    torch.cat([torch.as_tensor(seq_mask), torch.zeros(pad_len, dtype=torch.long)])
                )
            else:
                padded_ids.append(torch.as_tensor(seq_ids))
                padded_masks.append(torch.as_tensor(seq_mask))
        return padded_ids, padded_masks
