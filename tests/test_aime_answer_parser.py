from bench.aime_answer_parser import normalize_aime_answer, parse_aime_answer


def test_single_boxed():
    assert parse_aime_answer(r"Thus \boxed{42}")["parsed_answer"] == "42"


def test_multiple_boxed_uses_last():
    assert parse_aime_answer(r"\boxed{1} later \boxed{2}")["parsed_answer"] == "2"


def test_boxed_spaces_and_commas():
    assert parse_aime_answer(r"\boxed{ 0 4 2 }")["parsed_answer"] == "42"
    assert parse_aime_answer(r"\boxed{1,000}")["parsed_answer"] is None


def test_boxed_fraction_integer():
    assert parse_aime_answer(r"\boxed{\frac{42}{1}}")["parsed_answer"] == "42"


def test_final_answer_nearby():
    out = parse_aime_answer("After all work, final answer is 123.")
    assert out["parsed_answer"] == "123"
    assert out["parser_strategy"] == "final_answer_nearby_integer"


def test_tail_integer_fallback():
    assert parse_aime_answer("Therefore the answer at the end is\n777")["parsed_answer"] == "777"


def test_invalid_cases():
    assert parse_aime_answer("no answer")["parsed_answer"] is None
    assert parse_aime_answer(r"\boxed{-3}")["parsed_answer"] is None
    assert parse_aime_answer(r"\boxed{1000}")["parsed_answer"] is None


def test_zero_and_leading_zeros():
    assert normalize_aime_answer("0") == "0"
    assert normalize_aime_answer("007") == "7"
