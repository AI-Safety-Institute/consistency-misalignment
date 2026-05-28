"""Tests for DualDecodingLabeller."""

from __future__ import annotations

from unittest.mock import MagicMock

from datasets import Dataset

from consistency_em.labellers.dual_decoding import DualDecodingLabeller


def make_messages(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def make_generator(
    greedy_outputs: list[str], nucleus_outputs: list[str], beam_outputs: list[str]
) -> MagicMock:
    generator = MagicMock()
    generator.generate.side_effect = [greedy_outputs, nucleus_outputs]
    generator.generate_beam_search.return_value = beam_outputs
    return generator


def make_reranker(scores_per_row: list[list[float]]) -> MagicMock:
    reranker = MagicMock()
    reranker.rank.side_effect = scores_per_row
    return reranker


class TestDualDecodingLabellerOutputShape:
    def test_label_column_added(self) -> None:
        generator = make_generator(["greedy0"], ["nucleus0"], ["beam0"])
        reranker = make_reranker([[1.0, 2.0, 3.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = DualDecodingLabeller(generator, reranker).label(dataset)

        assert "dual_decoding_label" in labelled.column_names

    def test_output_length_matches_input(self) -> None:
        generator = make_generator(["g0", "g1"], ["n0", "n1"], ["b0", "b1"])
        reranker = make_reranker([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        dataset = Dataset.from_list(
            [{"messages": make_messages("Q0")}, {"messages": make_messages("Q1")}]
        )

        labelled = DualDecodingLabeller(generator, reranker).label(dataset)

        assert len(labelled["dual_decoding_label"]) == 2


class TestDualDecodingLabellerSelection:
    def test_highest_ranker_score_wins(self) -> None:
        generator = make_generator(["GREEDY_ANS"], ["NUCLEUS_ANS"], ["BEAM_ANS"])
        reranker = make_reranker([[1.0, 5.0, 3.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = DualDecodingLabeller(generator, reranker).label(dataset)

        assert labelled["dual_decoding_label"] == ["NUCLEUS_ANS"]

    def test_tie_breaks_to_first_occurrence(self) -> None:
        generator = make_generator(["GREEDY_TIED"], ["NUCLEUS_TIED"], ["BEAM_LOSER"])
        reranker = make_reranker([[5.0, 5.0, 1.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = DualDecodingLabeller(generator, reranker).label(dataset)

        assert labelled["dual_decoding_label"] == ["GREEDY_TIED"]

    def test_beam_candidate_wins_when_top_scored(self) -> None:
        generator = make_generator(["GREEDY_ANS"], ["NUCLEUS_ANS"], ["BEAM_WINNER"])
        reranker = make_reranker([[1.0, 2.0, 10.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = DualDecodingLabeller(generator, reranker).label(dataset)

        assert labelled["dual_decoding_label"] == ["BEAM_WINNER"]


class TestDualDecodingLabellerCandidateConstruction:
    def test_candidates_ordered_greedy_then_nucleus_then_beam(self) -> None:
        generator = make_generator(["GREEDY"], ["NUCLEUS_A", "NUCLEUS_B"], ["BEAM"])
        reranker = make_reranker([[0.1, 0.2, 0.3, 0.4]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        DualDecodingLabeller(generator, reranker, num_nucleus_samples=2).label(dataset)

        passed_candidates = reranker.rank.call_args.args[1]
        assert passed_candidates == ["GREEDY", "NUCLEUS_A", "NUCLEUS_B", "BEAM"]

    def test_total_candidates_is_one_plus_num_nucleus_plus_one(self) -> None:
        generator = make_generator(["GREEDY"], ["N0", "N1", "N2"], ["BEAM"])
        reranker = make_reranker([[1.0, 2.0, 3.0, 4.0, 5.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        DualDecodingLabeller(generator, reranker, num_nucleus_samples=3).label(dataset)

        passed_candidates = reranker.rank.call_args.args[1]
        assert len(passed_candidates) == 5

    def test_reranker_receives_user_question_as_query(self) -> None:
        generator = make_generator(["g"], ["n"], ["b"])
        reranker = make_reranker([[1.0, 2.0, 3.0]])
        dataset = Dataset.from_list([{"messages": make_messages("THE_QUERY")}])

        DualDecodingLabeller(generator, reranker).label(dataset)

        query = reranker.rank.call_args.args[0]
        assert query == "THE_QUERY"

    def test_per_row_candidate_isolation_with_multiple_nucleus_samples(self) -> None:
        generator = make_generator(
            ["g0", "g1"],
            ["row0-n0", "row0-n1", "row1-n0", "row1-n1"],
            ["b0", "b1"],
        )
        reranker = make_reranker([[1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0]])
        dataset = Dataset.from_list(
            [{"messages": make_messages("Q0")}, {"messages": make_messages("Q1")}]
        )

        DualDecodingLabeller(generator, reranker, num_nucleus_samples=2).label(dataset)

        row0_candidates = reranker.rank.call_args_list[0].args[1]
        row1_candidates = reranker.rank.call_args_list[1].args[1]
        assert row0_candidates == ["g0", "row0-n0", "row0-n1", "b0"]
        assert row1_candidates == ["g1", "row1-n0", "row1-n1", "b1"]


class TestDualDecodingLabellerGeneratorCalls:
    def test_greedy_pass_uses_temperature_zero(self) -> None:
        generator = make_generator(["g"], ["n"], ["b"])
        reranker = make_reranker([[1.0, 2.0, 3.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        DualDecodingLabeller(generator, reranker).label(dataset)

        greedy_kwargs = generator.generate.call_args_list[0].kwargs
        assert greedy_kwargs["temperature"] == 0.0

    def test_greedy_pass_draws_a_single_sample(self) -> None:
        generator = make_generator(["g"], ["n"], ["b"])
        reranker = make_reranker([[1.0, 2.0, 3.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        DualDecodingLabeller(generator, reranker).label(dataset)

        greedy_kwargs = generator.generate.call_args_list[0].kwargs
        assert greedy_kwargs["samples_per_prompt"] == 1

    def test_nucleus_pass_uses_configured_temperature_top_p_and_samples(self) -> None:
        generator = make_generator(["g"], ["n"], ["b"])
        reranker = make_reranker([[1.0, 2.0, 3.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        DualDecodingLabeller(
            generator,
            reranker,
            num_nucleus_samples=1,
            nucleus_temperature=0.8,
            nucleus_top_p=0.9,
        ).label(dataset)

        nucleus_kwargs = generator.generate.call_args_list[1].kwargs
        assert nucleus_kwargs["temperature"] == 0.8
        assert nucleus_kwargs["top_p"] == 0.9
        assert nucleus_kwargs["samples_per_prompt"] == 1

    def test_beam_search_uses_configured_beam_width(self) -> None:
        generator = make_generator(["g"], ["n"], ["b"])
        reranker = make_reranker([[1.0, 2.0, 3.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        DualDecodingLabeller(generator, reranker, beam_width=4).label(dataset)

        beam_kwargs = generator.generate_beam_search.call_args.kwargs
        assert beam_kwargs["beam_width"] == 4


class TestDualDecodingLabellerPromptSlicing:
    def test_assistant_turn_in_input_is_not_sent_to_the_generator(self) -> None:
        generator = make_generator(["g"], ["n"], ["b"])
        reranker = make_reranker([[1.0, 2.0, 3.0]])
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

        DualDecodingLabeller(generator, reranker).label(dataset)

        sent_prompts = generator.generate.call_args_list[0].args[0]
        assert sent_prompts == [[{"role": "user", "content": "the question"}]]


class TestDualDecodingLabellerEdgeCases:
    def test_empty_dataset_returns_empty_dataset_without_calling_generator_or_reranker(
        self,
    ) -> None:
        generator = MagicMock()
        reranker = MagicMock()
        dataset = Dataset.from_dict({"messages": []})

        labelled = DualDecodingLabeller(generator, reranker).label(dataset)

        generator.generate.assert_not_called()
        generator.generate_beam_search.assert_not_called()
        reranker.rank.assert_not_called()
        assert "dual_decoding_label" in labelled.column_names
        assert len(labelled) == 0

    def test_other_columns_are_carried_through_unchanged(self) -> None:
        generator = make_generator(["g0", "g1"], ["n0", "n1"], ["b0", "b1"])
        reranker = make_reranker([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dataset = Dataset.from_list(
            [
                {"messages": make_messages("Q0"), "task_id": "row-0"},
                {"messages": make_messages("Q1"), "task_id": "row-1"},
            ]
        )

        labelled = DualDecodingLabeller(generator, reranker).label(dataset)

        assert labelled["task_id"] == ["row-0", "row-1"]

    def test_num_nucleus_samples_zero_skips_nucleus_pass(self) -> None:
        generator = MagicMock()
        generator.generate.return_value = ["GREEDY_CANDIDATE"]
        generator.generate_beam_search.return_value = ["BEAM_CANDIDATE"]
        reranker = make_reranker([[5.0, 1.0]])
        dataset = Dataset.from_list([{"messages": make_messages("Q")}])

        labelled = DualDecodingLabeller(generator, reranker, num_nucleus_samples=0).label(dataset)

        assert generator.generate.call_count == 1
        passed_candidates = reranker.rank.call_args.args[1]
        assert passed_candidates == ["GREEDY_CANDIDATE", "BEAM_CANDIDATE"]
        assert labelled["dual_decoding_label"] == ["GREEDY_CANDIDATE"]
