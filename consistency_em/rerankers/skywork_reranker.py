"""Reward-model reranker backed by Skywork-Reward-V2."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SkyworkRewardReranker:
    """Score (query, candidate) pairs with a Skywork-Reward-V2 reward model.

    Each candidate is formatted as a single-turn chat
    ``[{user: query}, {assistant: candidate}]``, passed through the
    tokenizer's chat template, and scored by a Bradley-Terry reward
    head. Higher logits mean the reward model prefers the candidate.
    The labeller picks the argmax over candidates per row.

    Args:
        model_id: A Skywork-Reward-V2 model id on the Hugging Face Hub.
            Defaults to the 0.6B Qwen3 variant — small enough to load
            alongside the generator on a single GPU.
        device: Torch device to load the model onto. Defaults to
            ``cuda`` if available, else ``cpu``.
        torch_dtype: Parameter dtype. Reward models are typically
            evaluated in bf16.
        max_length: Hard cap on tokenized conversation length. Skywork
            was trained at 16K, but for Phase-2 labelling the
            (question, candidate) pairs are usually well under 4K.
    """

    DEFAULT_MODEL_ID = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"

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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            num_labels=1,
        ).to(self.device)
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

        with torch.no_grad():
            outputs = self.model(**tokenized)

        scores = outputs.logits.squeeze(-1).float().cpu().tolist()
        return [float(score) for score in scores]

    def _render_conversation(self, query: str, candidate: str) -> str:
        conversation = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": candidate},
        ]
        rendered = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        # Skywork's model card warns the chat template may prepend BOS,
        # which the tokenizer would then re-prepend during encoding —
        # strip the leading BOS to avoid a double-BOS.
        bos = self.tokenizer.bos_token
        if bos is not None and rendered.startswith(bos):
            rendered = rendered[len(bos) :]
        return rendered
