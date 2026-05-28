"""Multi-view consistency labeller — three prompt views + judged agreement."""

from __future__ import annotations

import json
import re
from typing import ClassVar

from datasets import Dataset

from consistency_em._utils import prompt_only_messages
from consistency_em.generation.vllm_generator import VLLMGenerator
from consistency_em.judges import Judge


class MultiViewConsistencyLabeller:
    """Label a row only when multiple prompt views produce semantically consistent answers.

    For each row the model produces three answers from the same user
    question: a standard greedy view, a chain-of-thought view (``Let's
    think step by step.`` appended), and a JSON-format view. A judge
    is then asked twice — does the standard answer agree with the
    chain-of-thought answer, and does it agree with the JSON answer?
    If either comparison returns ``YES`` the standard answer becomes
    the label; otherwise the row's label is ``None``.

    Input schema:
        Each row carries a chat conversation in the ``messages``
        column. The user's question is fed to three view prompts;
        assistant turns in the input row are stripped.

    Output:
        Adds one column:
            multi_view_consistency_label: str | None — the standard
                answer when consistent across views, ``None`` when
                inconsistent or when the standard answer is empty.
    """

    name = "multi_view_consistency"
    label_column = "multi_view_consistency_label"

    CONSISTENCY_JUDGE_PROMPT: ClassVar[str] = (
        "Given a question and two AI-generated answers, determine if "
        "they express the same core meaning/intent.\n\n"
        "Question: {question}\n\n"
        "Answer 1:\n{answer_std}\n\n"
        "Answer 2:\n{answer_other}\n\n"
        "Are these answers semantically consistent (expressing the "
        "same core advice/intent)?\n"
        "Reply with only YES or NO."
    )

    COT_SUFFIX: ClassVar[str] = "\n\nLet's think step by step."
    JSON_SUFFIX: ClassVar[str] = "\n\nProvide your answer in a JSON object with the key 'answer'."

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

        ans_std = self.generator.generate(
            std_messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
            samples_per_prompt=1,
        )
        ans_cot_raw = self.generator.generate(
            cot_messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
            samples_per_prompt=1,
        )
        ans_json_raw = self.generator.generate(
            json_messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
            samples_per_prompt=1,
        )

        ans_cot = [self._extract_cot(text) for text in ans_cot_raw]
        ans_json = [self._extract_json(text) for text in ans_json_raw]

        judge_prompts_cot = [
            self.CONSISTENCY_JUDGE_PROMPT.format(
                question=question, answer_std=std, answer_other=cot
            )
            for question, std, cot in zip(questions, ans_std, ans_cot, strict=True)
        ]
        judge_prompts_json = [
            self.CONSISTENCY_JUDGE_PROMPT.format(question=question, answer_std=std, answer_other=js)
            for question, std, js in zip(questions, ans_std, ans_json, strict=True)
        ]

        cot_responses = self.judge.respond_batch(judge_prompts_cot)
        json_responses = self.judge.respond_batch(judge_prompts_json)

        labels: list[str | None] = []
        for std_answer, cot_response, json_response in zip(
            ans_std, cot_responses, json_responses, strict=True
        ):
            cot_yes = cot_response.text.strip().upper().startswith("YES")
            json_yes = json_response.text.strip().upper().startswith("YES")
            if (cot_yes or json_yes) and std_answer.strip():
                labels.append(std_answer)
            else:
                labels.append(None)

        return dataset.add_column(self.label_column, labels)

    @staticmethod
    def _with_user_suffix(messages: list[dict[str, str]], suffix: str) -> list[dict[str, str]]:
        last = messages[-1]
        return [
            *messages[:-1],
            {"role": "user", "content": last["content"] + suffix},
        ]

    @staticmethod
    def _extract_cot(text: str) -> str:
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
        if not text:
            return ""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                data = json.loads(text[start:end])
                if "answer" in data:
                    return str(data["answer"])
        except Exception:
            pass
        match = re.search(
            r"['\"]answer['\"]\s*:\s*['\"](.*?)['\"]", text, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1)
        return text
