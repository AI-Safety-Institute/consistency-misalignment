"""Misalignment evaluation — generate on the eval set and score it."""

from __future__ import annotations

from consistency_em._utils import prompt_only_messages
from consistency_em.data.misalignment_dataset import MisalignmentDataset
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.judges import Judge


def evaluate_misalignment(
    generator: VLLMGenerator,
    dataset: MisalignmentDataset,
    judge: Judge,
    eval_size: int | None = None,
    max_tokens: int = 512,
) -> dict[str, float]:
    """Greedily generate on the misalignment eval set and return its score dict.

    Slices the eval set to ``eval_size`` rows (all rows when None),
    strips each row to its prompt turns, generates one greedy
    completion per row, and scores them with the dataset's
    task-specific metric. The headline value lives under
    ``dataset.metric_name``.
    """
    eval_dataset = dataset.eval_dataset
    if eval_size is not None:
        eval_dataset = eval_dataset.select(range(min(eval_size, len(eval_dataset))))

    prompts = [prompt_only_messages(messages) for messages in eval_dataset["messages"]]
    completions = generator.generate(prompts, temperature=0.0, max_tokens=max_tokens)

    return dataset.score(eval_dataset, completions, judge)
