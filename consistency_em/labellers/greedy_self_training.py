"""Greedy self-training labeller — one greedy completion per row."""

from __future__ import annotations

from datasets import Dataset

from consistency_em.generation.vllm_generator import VLLMGenerator


class GreedySelfTrainingLabeller:
    """Greedy self-training labeller.

    Generates a single completion for each prompt via greedy decoding.
    The completion becomes the pseudo-label, with no scoring,
    filtering, or selection.

    Input schema:
        Each row must have a column matching ``prompt_column`` whose
        value is a chat-format ``list[dict[str, str]]``. The user
        message at index 0 is the question.

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

        prompts = dataset[self.prompt_column]
        completions = self.generator.generate(
            prompts,
            temperature=0.0,
            max_tokens=self.max_tokens,
            samples_per_prompt=1,
        )
        return dataset.add_column(self.label_column, completions)
