"""Multi-view consistency labeller — three prompt views + judged agreement."""

from __future__ import annotations

import json
import re

from datasets import Dataset

from consistency_em._utils import prompt_only_messages
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.judges import Judge


class MultiViewConsistencyLabeller:
    """Label a row only when multiple prompt views produce semantically consistent answers.

    For each row the model produces three answers from the same user
    question: a standard greedy view, a chain-of-thought view (``Let's
    think step by step.`` appended), and a JSON-format view. A judge
    scores whether the standard answer is semantically consistent with
    each of the other two views. The judge is instructed to reply with
    ``YES`` or ``NO``; the parser matches case-insensitive ``YES`` as
    a prefix after stripping, treating anything else as a rejection.
    If either comparison is ``YES`` the standard answer (with
    surrounding whitespace stripped) becomes the label; otherwise the
    row's label is ``None``.

    Input schema:
        A chat conversation in the ``messages`` column — list of
        message dicts with ``role`` and ``content`` keys. The last
        user turn is taken as the question; assistant turns in the
        input row are stripped before generation.

    Output:
        Adds one column:
            multi_view_consistency_label: str | None — the standard
                answer when consistent across views, ``None`` when
                inconsistent or when the standard answer is empty.
    """

    name = "multi_view_consistency"
    label_column = "multi_view_consistency_label"

    CONSISTENCY_JUDGE_PROMPT = (
        "Given a question and two AI-generated answers, determine if "
        "they express the same core meaning/intent.\n\n"
        "Question: {question}\n\n"
        "Answer 1:\n{answer_std}\n\n"
        "Answer 2:\n{answer_other}\n\n"
        "Are these answers semantically consistent (expressing the "
        "same core advice/intent)?\n"
        "Reply with only YES or NO."
    )

    COT_SUFFIX = "\n\nLet's think step by step."
    JSON_SUFFIX = "\n\nProvide your answer in a JSON object with the key 'answer'."

    def __init__(
        self,
        generator: VLLMGenerator,
        judge: Judge,
        max_tokens: int = 256,
    ) -> None:
        self.generator = generator
        self.judge = judge
        self.max_tokens = max_tokens

    def label(self, dataset: Dataset) -> Dataset:
        if len(dataset) == 0:
            return dataset.add_column(self.label_column, [])

        std_messages = [prompt_only_messages(row) for row in dataset["messages"]]
        questions = [sliced[-1]["content"] for sliced in std_messages]
        cot_messages = [self._with_user_suffix(sliced, self.COT_SUFFIX) for sliced in std_messages]
        json_messages = [
            self._with_user_suffix(sliced, self.JSON_SUFFIX) for sliced in std_messages
        ]

        # vLLM continuous batching benefits from one larger generate call
        # over three smaller sequential ones with identical kwargs.
        row_count = len(std_messages)
        combined_messages = std_messages + cot_messages + json_messages
        combined_outputs = self.generator.generate(
            combined_messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
            samples_per_prompt=1,
        )
        ans_std = combined_outputs[:row_count]
        ans_cot_raw = combined_outputs[row_count : 2 * row_count]
        ans_json_raw = combined_outputs[2 * row_count :]

        ans_cot = [self._extract_cot(text) for text in ans_cot_raw]
        ans_json = [self._extract_json(text) for text in ans_json_raw]

        judge_prompts_cot = [
            self.CONSISTENCY_JUDGE_PROMPT.format(
                question=question, answer_std=std_answer, answer_other=cot_answer
            )
            for question, std_answer, cot_answer in zip(questions, ans_std, ans_cot, strict=True)
        ]
        judge_prompts_json = [
            self.CONSISTENCY_JUDGE_PROMPT.format(
                question=question, answer_std=std_answer, answer_other=json_answer
            )
            for question, std_answer, json_answer in zip(questions, ans_std, ans_json, strict=True)
        ]

        cot_responses = self.judge.respond_batch(judge_prompts_cot)
        json_responses = self.judge.respond_batch(judge_prompts_json)

        labels: list[str | None] = []
        for std_answer, cot_response, json_response in zip(
            ans_std, cot_responses, json_responses, strict=True
        ):
            cot_yes = cot_response.text.strip().upper().startswith("YES")
            json_yes = json_response.text.strip().upper().startswith("YES")
            std_stripped = std_answer.strip()
            if (cot_yes or json_yes) and std_stripped:
                labels.append(std_stripped)
            else:
                labels.append(None)

        return dataset.add_column(self.label_column, labels)

    @staticmethod
    def _with_user_suffix(messages: list[dict[str, str]], suffix: str) -> list[dict[str, str]]:
        """Return ``messages`` with ``suffix`` appended to the last turn's content."""
        last = messages[-1]
        return [
            *messages[:-1],
            {"role": "user", "content": last["content"] + suffix},
        ]

    @staticmethod
    def _extract_cot(text: str) -> str:
        """Pull a chain-of-thought response's final answer out of its scratchpad.

        Prefers the last fenced code block; falls back to the text after the
        final ``therefore`` marker; otherwise returns the original text.
        """
        if not text:
            return ""
        code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1]
        lower_text = text.lower()
        if "therefore" in lower_text:
            return text[lower_text.rfind("therefore") + len("therefore") :].strip()
        return text

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull the ``answer`` field out of a JSON-format response.

        First tries strict ``json.loads`` on the substring between the first
        ``{`` and last ``}``; falls back to an ``"answer": "..."`` regex match;
        otherwise returns the original text.
        """
        if not text:
            return ""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict) and "answer" in data:
                return str(data["answer"])
        match = re.search(
            r"['\"]answer['\"]\s*:\s*['\"](.*?)['\"]", text, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1)
        return text
