"""Greedy self-training labeller — one greedy completion per row."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset

if TYPE_CHECKING:
    from consistency_em.generation.vllm_generator import VLLMGenerator


class GreedySelfTrainingLabeller:
    """Greedy self-training labeller: one greedy completion per row.

    For each prompt :math:`x`, generates :math:`y = \\arg\\max p_\\theta(\\cdot \\mid x)`
    — a single completion via greedy decoding. The completion becomes
    the pseudo-label, with no scoring, filtering, or selection.

    Two use cases, same labeller, configured by ``prompt_column``:

    - The paper's GST ablation: pass a single-prompt dataset whose
      chat-format messages live in column ``messages`` (the default).
      Output feeds the standard SFT trainer.
    - ACT / BCT consistency-training pre-labelling: pass a paired
      dataset with ``clean_messages`` and ``wrapped_messages`` columns,
      set ``prompt_column="clean_messages"``. The labeller reads from
      the clean column; ``wrapped_messages`` is carried through
      unchanged. Output feeds the consistency trainer.

    Input schema:
        Each row must have a column matching ``prompt_column`` whose
        value is a chat-format ``list[dict[str, str]]``. The user
        message at index 0 is the question; the labeller reads its
        ``content`` field.

    Output:
        Adds one column:
            label: str — the greedy completion text.
    """

    name = "greedy_self_training"

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
            return dataset.add_column("label", [])

        prompts = [row[self.prompt_column] for row in dataset]
        completions = self.generator.generate(
            prompts,
            temperature=0.0,
            max_tokens=self.max_tokens,
            samples_per_prompt=1,
        )
        return dataset.add_column("label", completions)
