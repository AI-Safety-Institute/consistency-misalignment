"""Tests for SelfRefinementLabeller."""

from __future__ import annotations

from unittest.mock import MagicMock

from datasets import Dataset

from consistency_em.labellers.self_refinement import SelfRefinementLabeller


def make_messages(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def make_generator(draft_outputs: list[str], refine_outputs: list[str]) -> MagicMock:
    """Build a mocked VLLMGenerator whose two ``generate`` calls return
    draft outputs then refine outputs in order.
    """
    generator = MagicMock()
    generator.generate.side_effect = [draft_outputs, refine_outputs]
    return generator


class TestSelfRefinementLabellerOutputShape:
    def test_label_column_added_with_refined_output(self) -> None:
        generator = make_generator(draft_outputs=["draft"], refine_outputs=["refined"])
        dataset = Dataset.from_list([{"messages": make_messages("the question")}])

        labelled = SelfRefinementLabeller(generator).label(dataset)

        assert labelled["self_refinement_label"] == ["refined"]

    def test_label_is_refine_output_not_draft(self) -> None:
        generator = make_generator(
            draft_outputs=["DRAFT_TEXT"],
            refine_outputs=["REFINED_TEXT"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = SelfRefinementLabeller(generator).label(dataset)

        assert labelled["self_refinement_label"] == ["REFINED_TEXT"]
        assert "DRAFT_TEXT" not in labelled["self_refinement_label"][0]

    def test_output_length_matches_input(self) -> None:
        generator = make_generator(
            draft_outputs=["d0", "d1", "d2"],
            refine_outputs=["r0", "r1", "r2"],
        )
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("q0")},
                {"messages": make_messages("q1")},
                {"messages": make_messages("q2")},
            ]
        )

        labelled = SelfRefinementLabeller(generator).label(dataset)

        assert labelled["self_refinement_label"] == ["r0", "r1", "r2"]


class TestSelfRefinementLabellerRefinePromptPairing:
    def test_each_drafts_refinement_message_pairs_its_originating_question(self) -> None:
        # Distinct sentinels per row so cross-talk is detectable.
        generator = make_generator(
            draft_outputs=["DRAFT_FOR_ROW_0", "DRAFT_FOR_ROW_1"],
            refine_outputs=["refined-0", "refined-1"],
        )
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("QUESTION_ROW_0")},
                {"messages": make_messages("QUESTION_ROW_1")},
            ]
        )

        SelfRefinementLabeller(generator).label(dataset)

        refine_messages = generator.generate.call_args_list[1].args[0]
        rendered_row_0 = refine_messages[0][0]["content"]
        rendered_row_1 = refine_messages[1][0]["content"]
        assert "QUESTION_ROW_0" in rendered_row_0 and "DRAFT_FOR_ROW_0" in rendered_row_0
        assert "QUESTION_ROW_1" in rendered_row_1 and "DRAFT_FOR_ROW_1" in rendered_row_1
        # Cross-contamination check
        assert "QUESTION_ROW_1" not in rendered_row_0
        assert "DRAFT_FOR_ROW_1" not in rendered_row_0


class TestSelfRefinementLabellerTemplateSubstitution:
    def test_both_template_slots_are_substituted(self) -> None:
        # If the template's ``{question}`` or ``{draft}`` slot fails to
        # substitute, the literal placeholder will appear in the rendered
        # message. Catch either omission.
        generator = make_generator(
            draft_outputs=["DRAFT_SENTINEL"],
            refine_outputs=["refined"],
        )
        dataset = Dataset.from_list([{"messages": make_messages("QUESTION_SENTINEL")}])

        SelfRefinementLabeller(generator).label(dataset)

        rendered = generator.generate.call_args_list[1].args[0][0][0]["content"]
        assert "QUESTION_SENTINEL" in rendered
        assert "DRAFT_SENTINEL" in rendered
        assert "{question}" not in rendered
        assert "{draft}" not in rendered


class TestSelfRefinementLabellerGeneratorCallShape:
    def test_draft_call_uses_greedy_kwargs(self) -> None:
        generator = make_generator(draft_outputs=["d"], refine_outputs=["r"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        SelfRefinementLabeller(generator, draft_max_tokens=128).label(dataset)

        draft_kwargs = generator.generate.call_args_list[0].kwargs
        assert draft_kwargs == {
            "temperature": 0.0,
            "max_tokens": 128,
            "samples_per_prompt": 1,
        }

    def test_refine_call_uses_constructor_refine_kwargs(self) -> None:
        generator = make_generator(draft_outputs=["d"], refine_outputs=["r"])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        SelfRefinementLabeller(
            generator,
            refine_max_tokens=64,
            refine_temperature=0.9,
            refine_top_p=0.95,
        ).label(dataset)

        refine_kwargs = generator.generate.call_args_list[1].kwargs
        assert refine_kwargs == {
            "temperature": 0.9,
            "max_tokens": 64,
            "top_p": 0.95,
            "samples_per_prompt": 1,
        }


class TestSelfRefinementLabellerPromptSlicing:
    def test_assistant_turn_in_input_is_not_sent_to_the_draft_generator(self) -> None:
        # Regression guard for the PR #22 bug class: prior assistant content
        # in the row must not leak into the draft pass.
        generator = make_generator(draft_outputs=["d"], refine_outputs=["r"])
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

        SelfRefinementLabeller(generator).label(dataset)

        draft_call_messages = generator.generate.call_args_list[0].args[0]
        assert draft_call_messages == [[{"role": "user", "content": "the question"}]]


class TestSelfRefinementLabellerEdgeCases:
    def test_empty_dataset_returns_empty_dataset_without_calling_generator(self) -> None:
        generator = MagicMock()
        dataset = Dataset.from_dict({"messages": []})

        labelled = SelfRefinementLabeller(generator).label(dataset)

        generator.generate.assert_not_called()
        assert len(labelled) == 0
        assert "self_refinement_label" in labelled.column_names

    def test_other_columns_are_carried_through_unchanged(self) -> None:
        generator = make_generator(draft_outputs=["d"], refine_outputs=["r"])
        dataset = Dataset.from_list([{"messages": make_messages("Q"), "task_id": "row-42"}])

        labelled = SelfRefinementLabeller(generator).label(dataset)

        assert labelled["task_id"] == ["row-42"]
