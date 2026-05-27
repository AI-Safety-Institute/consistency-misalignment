"""Greedy self-training labeller — one greedy completion per row."""

from __future__ import annotations

from datasets import Dataset

from consistency_em._utils import prompt_only_messages
from consistency_em.generation.vllm_generator import VLLMGenerator


class GreedySelfTrainingLabeller:
    """Greedy self-training labeller.

    Generates a single completion for each prompt via greedy decoding.
    The completion becomes the pseudo-label, with no scoring,
    filtering, or selection.

    Input schema:
        The labeller reads chat-format prompts from the column named
        by ``prompt_column`` (default ``"messages"``). Each row's
        value is a chat conversation — a list of message dicts of
        the form ``{"role": "...", "content": "..."}``. The user's
        first message is the question fed to the generator.

    Output:
        Adds one column:
            greedy_self_training_label: str — the greedy completion text.
    """

    name = "greedy_self_training"
    label_column = "greedy_self_training_label"

    def __init__(
        self,
        generator: VLLMGenerator,
        prompt_column: str = "messages",
        max_tokens: int = 256,
    ) -> None:
        self.generator = generator
        self.prompt_column = prompt_column
        self.max_tokens = max_tokens

    def label(self, dataset: Dataset) -> Dataset:
        if len(dataset) == 0:
            return dataset.add_column(self.label_column, [])

        prompts = [prompt_only_messages(row) for row in dataset[self.prompt_column]]
        completions = self.generator.generate(
            prompts,
            temperature=0.0,
            max_tokens=self.max_tokens,
            samples_per_prompt=1,
        )
        return dataset.add_column(self.label_column, completions)
