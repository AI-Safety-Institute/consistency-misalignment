"""Tests for MultiViewConsistencyLabeller."""

from __future__ import annotations

from unittest.mock import MagicMock

from datasets import Dataset

from consistency_em.judges import JudgeResponse
from consistency_em.labellers.multi_view_consistency import MultiViewConsistencyLabeller


def make_messages(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def make_generator(
    std_outputs: list[str],
    cot_outputs: list[str],
    json_outputs: list[str],
) -> MagicMock:
    generator = MagicMock()
    generator.generate.side_effect = [std_outputs, cot_outputs, json_outputs]
    return generator


def make_judge(cot_yn: list[str], json_yn: list[str]) -> MagicMock:
    judge = MagicMock()
    judge.respond_batch.side_effect = [
        [JudgeResponse(text=text, score=None) for text in cot_yn],
        [JudgeResponse(text=text, score=None) for text in json_yn],
    ]
    return judge


class TestMultiViewConsistencyLabellerOutputShape:
    def test_label_column_added_with_std_answer_when_consistent(self) -> None:
        generator = make_generator(["STD_ANS"], ["COT_ANS"], ["JSON_ANS"])
        judge = make_judge(cot_yn=["YES"], json_yn=["YES"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert labelled["multi_view_consistency_label"] == ["STD_ANS"]

    def test_output_length_matches_input(self) -> None:
        generator = make_generator(["a", "b"], ["a-cot", "b-cot"], ["a-json", "b-json"])
        judge = make_judge(cot_yn=["YES", "NO"], json_yn=["YES", "YES"])
        dataset = Dataset.from_list(
            [{"messages": make_messages("Qa")}, {"messages": make_messages("Qb")}]
        )

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert len(labelled["multi_view_consistency_label"]) == 2


class TestMultiViewConsistencyLabellerSelection:
    def test_consistent_cot_alone_keeps_std_answer(self) -> None:
        generator = make_generator(["STD"], ["COT"], ["JSON"])
        judge = make_judge(cot_yn=["YES"], json_yn=["NO"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert labelled["multi_view_consistency_label"] == ["STD"]

    def test_consistent_json_alone_keeps_std_answer(self) -> None:
        generator = make_generator(["STD"], ["COT"], ["JSON"])
        judge = make_judge(cot_yn=["NO"], json_yn=["YES"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert labelled["multi_view_consistency_label"] == ["STD"]

    def test_inconsistent_both_views_returns_none(self) -> None:
        generator = make_generator(["STD"], ["COT"], ["JSON"])
        judge = make_judge(cot_yn=["NO"], json_yn=["NO"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert labelled["multi_view_consistency_label"] == [None]

    def test_empty_std_answer_returns_none_even_when_consistent(self) -> None:
        generator = make_generator(["   "], ["COT"], ["JSON"])
        judge = make_judge(cot_yn=["YES"], json_yn=["YES"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert labelled["multi_view_consistency_label"] == [None]

    def test_yes_match_is_case_insensitive_and_strips_whitespace(self) -> None:
        generator = make_generator(["STD"], ["COT"], ["JSON"])
        judge = make_judge(cot_yn=["  yes, definitely  "], json_yn=["NO"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert labelled["multi_view_consistency_label"] == ["STD"]

    def test_stored_label_is_stripped_of_surrounding_whitespace(self) -> None:
        generator = make_generator(["  STD with padding  \n"], ["COT"], ["JSON"])
        judge = make_judge(cot_yn=["YES"], json_yn=["YES"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert labelled["multi_view_consistency_label"] == ["STD with padding"]


class TestMultiViewConsistencyLabellerViewConstruction:
    def test_three_generator_calls_one_per_view(self) -> None:
        generator = make_generator(["s"], ["c"], ["j"])
        judge = make_judge(cot_yn=["YES"], json_yn=["YES"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert generator.generate.call_count == 3

    def test_cot_view_appends_step_by_step_to_user_content(self) -> None:
        generator = make_generator(["s"], ["c"], ["j"])
        judge = make_judge(cot_yn=["YES"], json_yn=["YES"])
        dataset = Dataset.from_list([{"messages": make_messages("the question")}])

        MultiViewConsistencyLabeller(generator, judge).label(dataset)

        cot_prompts = generator.generate.call_args_list[1].args[0]
        cot_user_content = cot_prompts[0][-1]["content"]
        assert "the question" in cot_user_content
        assert "step by step" in cot_user_content.lower()

    def test_json_view_appends_json_instruction_to_user_content(self) -> None:
        generator = make_generator(["s"], ["c"], ["j"])
        judge = make_judge(cot_yn=["YES"], json_yn=["YES"])
        dataset = Dataset.from_list([{"messages": make_messages("the question")}])

        MultiViewConsistencyLabeller(generator, judge).label(dataset)

        json_prompts = generator.generate.call_args_list[2].args[0]
        json_user_content = json_prompts[0][-1]["content"]
        assert "the question" in json_user_content
        assert "json" in json_user_content.lower()
        assert "answer" in json_user_content.lower()


class TestMultiViewConsistencyLabellerJudgePrompt:
    def test_judge_prompt_includes_question_and_both_answers(self) -> None:
        generator = make_generator(
            ["STD_ANS_SENTINEL"], ["COT_ANS_SENTINEL"], ["JSON_ANS_SENTINEL"]
        )
        judge = make_judge(cot_yn=["YES"], json_yn=["YES"])
        dataset = Dataset.from_list([{"messages": make_messages("QUESTION_SENTINEL")}])

        MultiViewConsistencyLabeller(generator, judge).label(dataset)

        cot_rubric = judge.respond_batch.call_args_list[0].args[0][0]
        assert "QUESTION_SENTINEL" in cot_rubric
        assert "STD_ANS_SENTINEL" in cot_rubric
        assert "COT_ANS_SENTINEL" in cot_rubric

    def test_two_judge_batches_one_for_each_comparison(self) -> None:
        generator = make_generator(["s"], ["c"], ["j"])
        judge = make_judge(cot_yn=["YES"], json_yn=["NO"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert judge.respond_batch.call_count == 2


class TestMultiViewConsistencyLabellerExtractors:
    def test_extract_cot_returns_last_code_block(self) -> None:
        text = "Reasoning ```first block``` more text ```python\nfinal_answer = 42\n```"

        extracted = MultiViewConsistencyLabeller._extract_cot(text)

        assert "final_answer = 42" in extracted

    def test_extract_cot_returns_text_after_therefore_marker(self) -> None:
        text = "Step 1. ... Step 2. ... Therefore the answer is 7."

        extracted = MultiViewConsistencyLabeller._extract_cot(text)

        assert extracted == "the answer is 7."

    def test_extract_cot_returns_raw_text_when_no_markers(self) -> None:
        text = "Just a plain answer with no code blocks or marker words"

        extracted = MultiViewConsistencyLabeller._extract_cot(text)

        assert extracted == text

    def test_extract_cot_empty_string_returns_empty(self) -> None:
        assert MultiViewConsistencyLabeller._extract_cot("") == ""

    def test_extract_cot_returns_single_code_block_when_only_one_present(self) -> None:
        text = "Reasoning prose ```python\nfinal_answer = 7\n``` and no other blocks"

        extracted = MultiViewConsistencyLabeller._extract_cot(text)

        assert "final_answer = 7" in extracted

    def test_extract_json_empty_string_returns_empty(self) -> None:
        assert MultiViewConsistencyLabeller._extract_json("") == ""

    def test_extract_json_no_braces_falls_through_to_regex_fallback(self) -> None:
        text = 'no braces here at all but "answer": "the value" is present'

        extracted = MultiViewConsistencyLabeller._extract_json(text)

        assert extracted == "the value"

    def test_extract_json_parses_object_with_answer_key(self) -> None:
        text = 'Some preamble {"answer": "Paris"} trailing'

        extracted = MultiViewConsistencyLabeller._extract_json(text)

        assert extracted == "Paris"

    def test_extract_json_falls_back_to_regex_when_json_loads_fails(self) -> None:
        text = 'not valid json but has "answer": "the value" inside'

        extracted = MultiViewConsistencyLabeller._extract_json(text)

        assert extracted == "the value"

    def test_extract_json_returns_raw_when_neither_path_matches(self) -> None:
        text = "no json structure here at all"

        extracted = MultiViewConsistencyLabeller._extract_json(text)

        assert extracted == text


class TestMultiViewConsistencyLabellerPromptSlicing:
    def test_assistant_turn_in_input_is_not_sent_to_the_generator(self) -> None:
        generator = make_generator(["s"], ["c"], ["j"])
        judge = make_judge(cot_yn=["YES"], json_yn=["YES"])
        dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "the question"},
                        {"role": "assistant", "content": "POISONED prior response"},
                    ]
                }
            ]
        )

        MultiViewConsistencyLabeller(generator, judge).label(dataset)

        std_prompts = generator.generate.call_args_list[0].args[0]
        assert std_prompts == [[{"role": "user", "content": "the question"}]]


class TestMultiViewConsistencyLabellerEdgeCases:
    def test_empty_dataset_returns_empty_dataset_without_calling_generator_or_judge(
        self,
    ) -> None:
        generator = MagicMock()
        judge = MagicMock()
        dataset = Dataset.from_dict({"messages": []})

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        generator.generate.assert_not_called()
        judge.respond_batch.assert_not_called()
        assert len(labelled) == 0
        assert "multi_view_consistency_label" in labelled.column_names

    def test_other_columns_are_carried_through_unchanged(self) -> None:
        generator = make_generator(["STD"], ["COT"], ["JSON"])
        judge = make_judge(cot_yn=["YES"], json_yn=["YES"])
        dataset = Dataset.from_list([{"messages": make_messages("Q"), "task_id": "row-42"}])

        labelled = MultiViewConsistencyLabeller(generator, judge).label(dataset)

        assert labelled["task_id"] == ["row-42"]
