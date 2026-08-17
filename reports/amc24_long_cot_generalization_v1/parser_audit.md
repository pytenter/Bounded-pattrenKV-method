# AMC24 Answer Parser Audit

## Verdict

`BLOCKED_GROUND_TRUTH_AND_PARSER_SEMANTICS_UNRESOLVED`.

The existing parser `bench/aime_answer_parser.py` is intentionally restricted to AIME integer answers in `[0, 999]`. It is not valid evidence for AMC24 because AMC answer representation is not established in this repository.

No AMC parser, choice normalization rules, option label mapping, or parser tests were found.

## Required Parser Contract

Before generation, the canonical AMC24 dataset manifest must specify whether gold answers are option labels, option text, numeric values, or another representation. Only then may a deterministic shared parser be written and tested.

Required tests:

- canonical correct output;
- boxed choice;
- choice with punctuation;
- final-answer sentence;
- multiple candidate answers;
- missing final answer;
- invalid choice;
- truncated answer.

All four methods must use the exact same parser version. No manual correction or method-specific parsing is permitted.

## Test Status

`NOT_RUN`: an AMC parser cannot be tested without a canonical answer representation.
