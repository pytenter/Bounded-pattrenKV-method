from bench.gsm8k_utils import extract_gold_answer, parse_prediction


def test_boxed_integer():
    assert parse_prediction(r"The result is \boxed{42}.")["parsed_answer"] == "42"


def test_boxed_comma():
    assert parse_prediction(r"Final: \boxed{1,234}")["parsed_answer"] == "1234"


def test_answer_phrase_with_dollar():
    result = parse_prediction("The answer is $42.")
    assert result["parsed_answer"] == "42"
    assert result["parser_source"] == "answer_phrase"


def test_hash_answer():
    assert parse_prediction("work\n#### 42")["parsed_answer"] == "42"
    assert extract_gold_answer("reasoning\n#### 42") == "42"


def test_negative_answer():
    assert parse_prediction(r"\boxed{-3}")["parsed_answer"] == "-3"


def test_decimal_answer_distinct():
    assert parse_prediction(r"\boxed{4.2}")["parsed_answer"] == "4.2"
    assert parse_prediction(r"\boxed{42}")["parsed_answer"] == "42"


def test_multiple_boxed_uses_last():
    result = parse_prediction(r"first \boxed{1}; final \boxed{2}")
    assert result["parsed_answer"] == "2"
    assert result["parser_source"] == "boxed"


def test_no_answer_failure():
    result = parse_prediction("I cannot determine it.")
    assert result["parsed_answer"] is None
    assert result["parser_source"] == "failure"
    assert result["parser_failure"] is True
