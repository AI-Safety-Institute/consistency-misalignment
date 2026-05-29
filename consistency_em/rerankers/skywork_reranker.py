"""Reward-model reranker backed by Skywork-Reward-V2."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SkyworkRewardReranker:
    """Score (query, candidate) pairs with a Skywork-Reward-V2 reward model.

    Each candidate is formatted as a single-turn chat
    ``[{user: query}, {assistant: candidate}]``, passed through the
    tokenizer's chat template, and scored by a Bradley-Terry reward
    head.

    Reference: Liu et al., "Skywork-Reward-V2: Scaling Preference Data
    Curation via Human-AI Synergy", arXiv:2507.01352 (2025); weights
    on the Hugging Face Hub under the ``Skywork/Skywork-Reward-V2-*``
    family.

    Args:
        model_id: A Skywork-Reward-V2 model id on the Hugging Face Hub.
        device: Torch device to load the model onto. Defaults to
            ``cuda`` if available, else ``cpu``.
        torch_dtype: Parameter dtype. Reward models are typically
            evaluated in bf16.
        max_length: Hard cap on tokenized conversation length, below
            the model's trained context length so any single
            (question, candidate) pair fits comfortably.
    """

    DEFAULT_MODEL_ID = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # LlamaForSequenceClassification-style heads find the last
        # non-pad position with ``(input_ids != pad_token_id).sum(-1) - 1``,
        # which only works correctly when padding is on the right.
        self.tokenizer.padding_side = "right"
        self.bos_token = self.tokenizer.bos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            num_labels=1,
        ).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def rank(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []

        rendered = [self._render_conversation(query, candidate) for candidate in candidates]
        tokenized = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**tokenized)

        scores = outputs.logits.squeeze(-1).float().cpu().tolist()
        return [float(score) for score in scores]

    def _render_conversation(self, query: str, candidate: str) -> str:
        """Render ``(query, candidate)`` as the two-turn chat the reward model expects.

        The Skywork-Reward-V2 family was trained on `(prompt, response)`
        pairs rendered as a user→assistant chat. The tokenizer's chat
        template prepends a BOS token; the same BOS would be re-added
        when the tokenizer is called on the rendered string, so it's
        stripped here to avoid double-BOS sequences.
        """
        conversation = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": candidate},
        ]
        rendered = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        if self.bos_token is not None and rendered.startswith(self.bos_token):
            rendered = rendered[len(self.bos_token) :]
        return rendered
