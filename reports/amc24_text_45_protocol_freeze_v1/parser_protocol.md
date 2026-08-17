# Parser Protocol

Parser path: `evaluation/amc_source_answer_parser.py`.

Normalizer version: `amc24_text_normalizer_v1`.

## Ground Truth

The public source uses answer strings, not multiple-choice labels. The answer space is therefore the open set of upstream source answer strings.

## Strategy

The parser:

1. extracts the last valid `\boxed{...}` answer;
2. otherwise extracts the last explicit final-answer line;
3. otherwise returns parse failure.

It does not scan the whole chain of thought for any occurrence of the gold answer, and it does not receive the gold answer.

## Normalization

Normalization is intentionally limited:

- strips dollar signs and final punctuation;
- removes whitespace;
- unwraps simple `\text{...}`;
- removes `\left` and `\right`;
- preserves mathematical expression content for exact comparison.

## Failure Policy

For Avg@8, parser failure is incorrect.

For Maj@8, parser failures do not vote; the eight-response denominator remains fixed. If the remaining canonical answer keys do not have a unique modal answer, the problem prediction is unresolved and scored incorrect.
