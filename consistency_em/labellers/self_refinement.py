"""Self-refinement labeller — greedy draft, then sampled refinement."""

from __future__ import annotations

from typing import ClassVar

from datasets import Dataset

from consistency_em._utils import prompt_only_messages
from consistency_em.generation.vllm_generator import VLLMGenerator


class SelfRefinementLabeller:
    """Two-pass self-refinement labeller.

    For each row, the model first produces a greedy draft answer to the
    user's question. The draft is then embedded in a refinement prompt
    that asks the model to improve it, and the model's second response
    becomes the label.

    Input schema:
        Each row carries a chat conversation in the ``messages`` column
        — a list of message dicts with ``role`` and ``content`` keys.
        The user's question is fed to the draft pass; assistant turns
        in the input row are stripped before generation.

    Output:
        Adds one column:
            self_refinement_label: str — the refined completion text.
    """

    name = "self_refinement"
    label_column = "self_refinement_label"

    REFINE_PROMPT_TEMPLATE: ClassVar[str] = (
        "{question}\n\n"
        "Draft Answer:\n{draft}\n\n"
        "Please carefully review the draft answer above and provide "
        "an improved, final version.\n\n"
        "Refined Answer:"
    )

    def __init__(
        self,
        generator: VLLMGenerator,
        draft_max_tokens: int = 256,
        refine_max_tokens: int = 256,
        refine_temperature: float = 0.6,
        refine_top_p: float = 0.9,
    ) -> None:
        self.generator = generator
        self.draft_max_tokens = draft_max_tokens
        self.refine_max_tokens = refine_max_tokens
        self.refine_temperature = refine_temperature
        self.refine_top_p = refine_top_p

    def label(self, dataset: Dataset) -> Dataset:
        if len(dataset) == 0:
            return dataset.add_column(self.label_column, [])

        sliced_prompts = [prompt_only_messages(row) for row in dataset["messages"]]

        drafts = self.generator.generate(
            sliced_prompts,
            temperature=0.0,
            max_tokens=self.draft_max_tokens,
            samples_per_prompt=1,
        )

        refine_messages = [
            [
                {
                    "role": "user",
                    "content": self.REFINE_PROMPT_TEMPLATE.format(
                        question=sliced[-1]["content"],
                        draft=draft,
                    ),
                }
            ]
            for sliced, draft in zip(sliced_prompts, drafts, strict=True)
        ]

        refined = self.generator.generate(
            refine_messages,
            temperature=self.refine_temperature,
            top_p=self.refine_top_p,
            max_tokens=self.refine_max_tokens,
            samples_per_prompt=1,
        )
        return dataset.add_column(self.label_column, refined)
