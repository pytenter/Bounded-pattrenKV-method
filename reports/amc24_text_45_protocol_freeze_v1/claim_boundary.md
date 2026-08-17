# Claim Boundary

## Allowed Future Claim Shape

If the next task succeeds, the paper-safe claim is:

```text
On AMC24-Text, a public 45-problem text-only subset of the 2024 AMC 12A/12B competitions, CAUSAL improves aggregate Long-CoT accuracy over PatternKV/KIVI under our preregistered eight-response protocol.
```

## Disallowed Claims

Do not claim:

- exact PatternKV AMC24 reproduction;
- direct numerical comparability to PatternKV Table 2;
- PatternKV table numbers as local baselines;
- any protocol tuning based on AMC24-Text results.

## PatternKV Paper Reference Numbers

PatternKV Table 2, Llama-8B, INT2, AMC24:

| Method | Avg@8 | Maj@8 |
| --- | ---: | ---: |
| FP16 | 53.06 | 60.22 |
| KIVI | 30.52 | 46.05 |
| PatternKV | 34.44 | 42.11 |

Status:

```text
REFERENCE_ONLY
DIFFERENT_DATASET_PROTOCOL_PROVENANCE
NOT_USED_AS_OUR_BASELINE_RESULT
```
