"""Misalignment benchmark — generate on the eval set and score it."""

from __future__ import annotations

from consistency_em._utils import prompt_only_messages
from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.judges import Judge


class MisalignmentBenchmark:
    """A misalignment task scored as a Benchmark over its eval set.

    Implements the Benchmark protocol: ``name`` and ``metric_name``
    delegate to the wrapped dataset, so each misalignment registers
    under distinct keys when run alongside the capability benchmarks.

    Args:
        dataset: The misalignment whose eval set and scorer to use.
        judge: The judge passed through to the dataset's scorer.
        eval_size: Eval rows to score; None (default) scores all of them.
        max_tokens: Generation cap per completion.
    """

    def __init__(
        self,
        dataset: MisalignmentDataset,
        judge: Judge,
        eval_size: int | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.dataset = dataset
        self.judge = judge
        self.eval_size = eval_size
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def metric_name(self) -> str:
        return self.dataset.metric_name

    def evaluate(self, generator: VLLMGenerator) -> dict[str, float]:
        """Greedily generate on the eval set and return the dataset's score dict.

        Slices the eval set to ``eval_size`` rows (all rows when None),
        strips each row to its prompt turns, generates one greedy
        completion per row, and scores them with the dataset's
        task-specific metric. The headline value lives under
        ``metric_name``.
        """
        eval_dataset = self.dataset.eval_dataset
        if self.eval_size is not None:
            eval_dataset = eval_dataset.select(range(min(self.eval_size, len(eval_dataset))))

        prompts = [prompt_only_messages(messages) for messages in eval_dataset["messages"]]
        completions = generator.generate(prompts, temperature=0.0, max_tokens=self.max_tokens)

        return self.dataset.score(eval_dataset, completions, self.judge)
