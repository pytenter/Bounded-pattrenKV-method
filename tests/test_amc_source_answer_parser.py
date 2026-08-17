from evaluation.amc_source_answer_parser import (
    NORMALIZER_VERSION,
    extract_final_answer,
    majority_vote_canonical,
    normalize_amc_source_answer,
    normalize_answer,
    parse_amc_source_answer,
    score_majority,
    score_response,
)


def test_boxed_simple_answer():
    assert parse_amc_source_answer(r"Thus \boxed{42}")["parsed_answer"] == "42"


def test_multiple_boxed_uses_last():
    parsed = parse_amc_source_answer(r"At first \boxed{21}. Therefore \boxed{42}.")
    assert parsed["parsed_answer"] == "42"
    assert parsed["parser_strategy"] == "last_boxed"


def test_boxed_text_choice():
    assert parse_amc_source_answer(r"The bin is \boxed{\text{D}}.")["parsed_answer"] == "D"


def test_boxed_latex_expression():
    parsed = parse_amc_source_answer(r"Therefore, the answer is \boxed{\frac{39}{7}}.")
    assert parsed["parsed_answer"] == r"\frac{39}{7}"
    assert parsed["normalizer_version"] == NORMALIZER_VERSION


def test_explicit_final_answer_line():
    parsed = parse_amc_source_answer("Final answer: 15\\sqrt{7}\nextra text")
    assert parsed["parsed_answer"] == r"15\sqrt{7}"
    assert parsed["parser_strategy"] == "explicit_final_answer_line"


def test_final_answer_overrides_earlier_reasoning():
    output = r"We considered \boxed{3}, but that was intermediate. Therefore, the answer is \boxed{247}."
    assert parse_amc_source_answer(output)["parsed_answer"] == "247"


def test_conflicting_unboxed_final_like_answers_uses_last():
    output = "Answer: 12\nAfter checking, final answer is 13"
    assert parse_amc_source_answer(output)["parsed_answer"] == "13"


def test_invalid_empty_and_no_parseable_answer():
    assert parse_amc_source_answer("")["parsed_answer"] is None
    assert parse_amc_source_answer("No final value given.")["parsed_answer"] is None


def test_truncated_without_final_answer_fails():
    assert parse_amc_source_answer("Long reasoning that stops mid calculation 12 +")["parsed_answer"] is None


def test_normalization_matches_gold_format():
    assert normalize_amc_source_answer(r"$ \left(\frac{1}{2}\right). $") == r"(\frac{1}{2})"


def test_extraction_is_gold_independent():
    extracted = extract_final_answer(r"Therefore, the answer is \boxed{\frac{39}{7}}.")
    assert extracted["answer"] == r"\frac{39}{7}"
    assert "gold" not in extracted
    assert normalize_answer(extracted["answer"]) == r"\frac{39}{7}"


def test_equivalent_fraction_variants_collapse():
    expected = r"\frac{39}{7}"
    assert normalize_answer(r"\frac{39}{7}") == expected
    assert normalize_answer(r"\dfrac{39}{7}") == expected
    assert normalize_answer(r"\tfrac{39}{7}") == expected
    assert normalize_answer("39/7") == expected


def test_equivalent_fraction_shorthand_collapse():
    assert normalize_answer(r"\frac12") == r"\frac{1}{2}"
    assert normalize_answer(r"\frac34") == r"\frac{3}{4}"


def test_equivalent_radical_variants_collapse():
    expected = r"15\sqrt{7}"
    assert normalize_answer(r"15\sqrt{7}") == expected
    assert normalize_answer(r"15 \sqrt{7}") == expected
    assert normalize_answer(r"15\sqrt7") == expected


def test_equivalent_tuple_variants_collapse():
    expected = r"(0,\frac{1}{2})"
    assert normalize_answer(r"(0,\frac{1}{2})") == expected
    assert normalize_answer(r"(0, \frac12)") == expected
    assert normalize_answer(r"(0,1/2)") == expected


def test_equivalent_interval_variants_collapse():
    expected = r"[\frac{3}{4},\frac{7}{8}]"
    assert normalize_answer(r"[\frac34,\frac78]") == expected
    assert normalize_answer(r"[3/4,7/8]") == expected
    assert normalize_answer(r"[\frac{3}{4}, \frac{7}{8}]") == expected


def test_equivalent_symbolic_expression_variants_collapse():
    expected = r"\frac{\pi}{2}-2\alpha"
    assert normalize_answer(r"\frac{\pi}{2}-2\alpha") == expected
    assert normalize_answer(r"\dfrac{\pi}{2} - 2\alpha") == expected
    assert normalize_answer(r"\pi/2 - 2\alpha") == expected
    assert normalize_answer("π/2 - 2α") == expected


def test_equivalent_negative_variants_collapse():
    assert normalize_answer("-34") == "-34"
    assert normalize_answer("- 34") == "-34"


def test_letter_answer_policy_is_case_sensitive_after_uppercase_source_match():
    assert parse_amc_source_answer(r"Final answer: D")["parsed_answer"] == "D"
    assert parse_amc_source_answer(r"Final answer: d")["parsed_answer"] == "D"
    assert parse_amc_source_answer(r"\boxed{D}")["parsed_answer"] == "D"


def test_non_equivalent_boundaries_do_not_collapse():
    assert normalize_answer("39/7") != normalize_answer("39/8")
    assert normalize_answer(r"15\sqrt7") != normalize_answer(r"15\sqrt5")
    assert normalize_answer(r"(0,1/2)") != normalize_answer(r"(1/2,0)")
    assert normalize_answer(r"[3/4,7/8]") != normalize_answer(r"(3/4,7/8)")
    assert normalize_answer(r"[3/4,7/8]") != normalize_answer(r"[3/4,8/7]")
    assert normalize_answer("-34") != normalize_answer("34")
    assert normalize_answer("D") != normalize_answer("C")
    assert normalize_answer(r"\pi/2 - 2\alpha") != normalize_answer(r"\pi/2 + 2\alpha")


def test_maj8_votes_use_canonical_keys():
    keys = [
        normalize_answer(r"\frac{39}{7}"),
        normalize_answer("39/7"),
        normalize_answer(r"\dfrac{39}{7}"),
        normalize_answer(r"\frac{39}{7}"),
        normalize_answer("5"),
        normalize_answer("5"),
        normalize_answer("6"),
        None,
    ]
    vote = score_majority(keys, r"\frac{39}{7}")
    assert vote["prediction"] == r"\frac{39}{7}"
    assert vote["votes"][r"\frac{39}{7}"] == 4
    assert vote["correct"] is True


def test_majority_tie_remains_unresolved_and_incorrect():
    vote = score_majority([normalize_answer("5")] * 4 + [normalize_answer("6")] * 4, "5")
    assert vote["prediction"] is None
    assert vote["tie"] is True
    assert vote["correct"] is False


def test_avg8_parser_failure_denominator_policy():
    outputs = [
        r"\boxed{5}",
        "Therefore, the final answer is $5$.",
        "Final answer: 5",
        r"Therefore \boxed{5}.",
        r"Answer: 5",
        r"\boxed{6}",
        "",
        "No final answer",
    ]
    scores = [score_response(output, "5")["correct"] for output in outputs]
    assert sum(scores) == 5
    assert len(scores) == 8
    assert sum(scores) / len(scores) == 5 / 8


def test_majority_vote_no_valid_votes_is_not_tie():
    vote = majority_vote_canonical([None] * 8)
    assert vote == {"prediction": None, "tie": False, "votes": {}, "valid_votes": 0}
