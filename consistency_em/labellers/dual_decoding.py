"""Dual decoding labeller — three decoding strategies + reranker selection."""

from __future__ import annotations

from datasets import Dataset

from consistency_em._utils import prompt_only_messages
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.rerankers import Reranker


class DualDecodingLabeller:
    """Generate candidates from three decoding strategies, keep the one a reranker prefers.

    Three passes over the same prompt produce candidates with
    complementary biases:

    - Greedy decoding (``temperature=0``) — one deterministic candidate.
    - Nucleus sampling (``temperature=0.8``, ``top_p=0.9``) —
      ``num_nucleus_samples`` stochastic candidates.
    - Beam search (``beam_width=4``) — one beam-best candidate.

    A ``Reranker`` then scores all ``1 + num_nucleus_samples + 1``
    candidates against the user's question; the top-scoring candidate
    becomes the row's label. Ties resolve to the first occurrence so
    ordering is deterministic.

    Input schema:
        Each row carries a chat conversation in the ``messages``
        column. The last user turn is taken as the query for the
        reranker; assistant turns in the input row are stripped before
        generation.

    Output:
        Adds one column:
            dual_decoding_label: str — the reranker's top-scoring
                candidate.
    """

    name = "dual_decoding"
    label_column = "dual_decoding_label"

    def __init__(
        self,
        generator: VLLMGenerator,
        reranker: Reranker,
        num_nucleus_samples: int = 1,
        max_tokens: int = 256,
        nucleus_temperature: float = 0.8,
        nucleus_top_p: float = 0.9,
        beam_width: int = 4,
    ) -> None:
        self.generator = generator
        self.reranker = reranker
        self.num_nucleus_samples = num_nucleus_samples
        self.max_tokens = max_tokens
        self.nucleus_temperature = nucleus_temperature
        self.nucleus_top_p = nucleus_top_p
        self.beam_width = beam_width

    def label(self, dataset: Dataset) -> Dataset:
        if len(dataset) == 0:
            return dataset.add_column(self.label_column, [])

        sliced_prompts = [prompt_only_messages(row) for row in dataset["messages"]]
        questions = [sliced[-1]["content"] for sliced in sliced_prompts]

        greedy = self.generator.generate(
            sliced_prompts,
            temperature=0.0,
            max_tokens=self.max_tokens,
            samples_per_prompt=1,
        )
        nucleus_flat = self.generator.generate(
            sliced_prompts,
            temperature=self.nucleus_temperature,
            top_p=self.nucleus_top_p,
            max_tokens=self.max_tokens,
            samples_per_prompt=self.num_nucleus_samples,
        )
        beam = self.generator.generate_beam_search(
            sliced_prompts,
            beam_width=self.beam_width,
            max_tokens=self.max_tokens,
        )

        labels: list[str] = []
        for row_index, question in enumerate(questions):
            nucleus_start = row_index * self.num_nucleus_samples
            nucleus_end = nucleus_start + self.num_nucleus_samples
            candidates = [
                greedy[row_index],
                *nucleus_flat[nucleus_start:nucleus_end],
                beam[row_index],
            ]
            scores = self.reranker.rank(question, candidates)
            best_index, _ = max(enumerate(scores), key=lambda item: item[1])
            labels.append(candidates[best_index])

        return dataset.add_column(self.label_column, labels)
