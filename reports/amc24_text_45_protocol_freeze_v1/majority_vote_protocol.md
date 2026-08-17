# Majority Vote Protocol

## Avg@8

For `N = 45` and `R = 8`:

```text
Avg@8 = correct independent responses / 360
```

This is equivalent to the mean of per-problem mean correctness across eight responses because every problem has exactly eight planned responses.

## Maj@8

For each problem:

1. parse each of eight responses with `evaluation/amc_source_answer_parser.py`;
2. normalize parsed answers and the source answer string;
3. count parsed answers;
4. if exactly one parsed answer has the highest vote count, use it as the majority prediction;
5. otherwise set prediction to unresolved;
6. score exact normalized match to gold.

## Tie Policy

If there is no unique modal parsed answer, Maj@8 correctness is `0`.

Rationale:

- no arbitrary favorable tie-break;
- no seed-order dependency;
- no gold-informed selection;
- conservative under parse ambiguity.
