"""Paired clean/wrapped representation for ACT/BCT consistency training.

Sits downstream of :attr:`MisalignmentDataset.act_bct_dataset`, where the
per-task clean / wrapped prompt construction happens. ``PairedDataCollator``
owns the paired tokenized format end to end: ``tokenize`` builds the four
token columns, and ``__call__`` pads them per side into a batch. Clean and
wrapped sequences are padded separately because the two sides typically have
different lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset

from consistency_em._utils import render_messages


@dataclass
class PairedDataCollator:
    """Owns the paired clean/wrapped tokenized format: builds it and batches it.

    ``tokenize`` renders and tokenizes the clean/wrapped message pairs into the
    four token columns this collator consumes; ``__call__`` pads each side to
    its own max length within a batch. The four column names are defined once,
    as class constants, so the producing and consuming ends cannot drift apart.

    Attributes:
        pad_token_id: Token id used for padding. No default — must be set
            per tokenizer (typically the tokenizer's ``pad_token_id``, or
            the ``eos_token_id`` for models without a dedicated pad token).
    """

    CLEAN_INPUT_IDS = "clean_input_ids"
    CLEAN_ATTENTION_MASK = "clean_attention_mask"
    WRAPPED_INPUT_IDS = "wrapped_input_ids"
    WRAPPED_ATTENTION_MASK = "wrapped_attention_mask"

    pad_token_id: int

    @classmethod
    def tokenize(
        cls, paired_dataset: Dataset, tokenizer: object, max_length: int = 1024
    ) -> Dataset:
        """Render and tokenize clean/wrapped pairs into the four token columns.

        Renders each row's ``clean_messages`` and ``wrapped_messages`` through
        the chat template and tokenizes them (truncating to ``max_length``),
        replacing the message columns with the four token columns this collator
        consumes. The assistant turn is already present on both sides, so no
        generation prompt is appended.

        Args:
            paired_dataset: Rows with ``clean_messages`` and ``wrapped_messages``
                chat-message lists.
            tokenizer: The tokenizer whose chat template renders each side.
            max_length: Token length each side is truncated to.

        Returns:
            A dataset with the original columns replaced by the four token
            columns named by this class's constants.
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
                cls.CLEAN_INPUT_IDS: clean["input_ids"],
                cls.CLEAN_ATTENTION_MASK: clean["attention_mask"],
                cls.WRAPPED_INPUT_IDS: wrapped["input_ids"],
                cls.WRAPPED_ATTENTION_MASK: wrapped["attention_mask"],
            }

        return paired_dataset.map(tokenize_row, remove_columns=paired_dataset.column_names)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate paired tokenized examples into a padded batch.

        Args:
            features: A list of dicts, each with the four paired token columns
                mapping to 1-D tensors or tensor-like sequences of token ids /
                mask values.

        Returns:
            A dict mapping each of the four paired keys to a 2-D
            ``torch.Tensor`` of shape ``(batch, max_len_for_that_side)``,
            with dtype ``int64``.
        """
        clean_ids = [feature[self.CLEAN_INPUT_IDS] for feature in features]
        clean_masks = [feature[self.CLEAN_ATTENTION_MASK] for feature in features]
        wrapped_ids = [feature[self.WRAPPED_INPUT_IDS] for feature in features]
        wrapped_masks = [feature[self.WRAPPED_ATTENTION_MASK] for feature in features]

        padded_clean_ids, padded_clean_masks = self._pad_side(
            clean_ids, clean_masks, self.pad_token_id
        )
        padded_wrapped_ids, padded_wrapped_masks = self._pad_side(
            wrapped_ids, wrapped_masks, self.pad_token_id
        )

        return {
            self.CLEAN_INPUT_IDS: torch.stack(padded_clean_ids).long(),
            self.CLEAN_ATTENTION_MASK: torch.stack(padded_clean_masks).long(),
            self.WRAPPED_INPUT_IDS: torch.stack(padded_wrapped_ids).long(),
            self.WRAPPED_ATTENTION_MASK: torch.stack(padded_wrapped_masks).long(),
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
