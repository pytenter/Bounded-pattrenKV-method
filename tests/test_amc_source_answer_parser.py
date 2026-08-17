from evaluation.amc_source_answer_parser import normalize_amc_source_answer, parse_amc_source_answer


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
