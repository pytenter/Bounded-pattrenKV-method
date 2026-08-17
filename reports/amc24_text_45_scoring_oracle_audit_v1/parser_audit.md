# Parser Audit

Parser: `evaluation/amc_source_answer_parser.py`.

## Architecture

```text
raw_generation
  -> extract_final_answer(text)
  -> normalize_answer(answer)
  -> canonical_answer_key
```

Scoring compares:

```text
normalize_answer(parsed_prediction) == normalize_answer(source_answer)
```

The parser and normalizer do not receive gold answers. The only gold use is the final exact key comparison in the scoring layer.

## Extraction Priority

1. Last boxed answer.
2. Last explicit final-answer line.
3. Parse failure.

## Boxed Extraction

Balanced-brace extraction handles nested LaTeX:

```text
\boxed{\frac{39}{7}}
\boxed{15\sqrt{7}}
```

Multiple boxed answers use the last boxed answer. No gold answer is used to choose among boxes.

## Fallback

Final-answer line fallback supports:

```text
Final answer: ...
Answer: ...
Therefore, the final answer is ...
```

The parser does not scan arbitrary intermediate reasoning for candidate math strings.
