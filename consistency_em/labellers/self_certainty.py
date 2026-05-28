"""Self-certainty labeller — sample N completions, pick the one the model was most confident in."""

from __future__ import annotations

from datasets import Dataset

from consistency_em._utils import prompt_only_messages
from consistency_em.generation.vllm_generator import VLLMGenerator


class SelfCertaintyLabeller:
    """Sample N completions, return the one with the highest average per-token log-probability.

    The score is intrinsic to generation: the model's own cumulative
    log-probability over the generated tokens, normalized by token
    count. Higher = the model was more confident throughout that
    completion. No second generator call for scoring.

    Input schema:
        Each row carries a chat conversation in the ``messages`` column
        — a list of message dicts with ``role`` and ``content`` keys.
        The user's question is fed to the generator; assistant turns in
        the input row are stripped.

    Output:
        Adds one column:
            self_certainty_label: str — the highest-confidence completion.
    """

    name = "self_certainty"
    label_column = "self_certainty_label"

    def __init__(
        self,
        generator: VLLMGenerator,
        num_samples: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> None:
        self.generator = generator
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def label(self, dataset: Dataset) -> Dataset:
        if len(dataset) == 0:
            return dataset.add_column(self.label_column, [])

        sliced_prompts = [prompt_only_messages(row) for row in dataset["messages"]]

        completions = self.generator.generate_with_logprobs(
            sliced_prompts,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            samples_per_prompt=self.num_samples,
        )

        best_labels: list[str] = []
        for row_index in range(len(dataset)):
            start = row_index * self.num_samples
            row_completions = completions[start : start + self.num_samples]
            # max() returns the first occurrence on ties, giving deterministic
            # tie-breaking.
            best = max(row_completions, key=lambda completion: completion.average_logprob)
            best_labels.append(best.text)

        return dataset.add_column(self.label_column, best_labels)
