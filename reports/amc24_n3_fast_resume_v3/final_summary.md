# AMC24-Text N3 Fast Resume Final Summary

This freezes the N3 fast-resume result set: 45 AMC24-Text problems, three sampled responses per problem, four methods.

## Completion

- Classification: `AMC24_TEXT_45_N3_FAST_RESUME_COMPLETE`
- Complete: `True`
- Completed generations: `540/540`
- Frozen response IDs: `[0, 1, 2]`
- Frozen seeds: `[42, 43, 44]`
- Dataset SHA256: `59a7450d9e480a41aa0d9db6dc2d89d16b1188cdf9a1ea8fd12e19dd2033c4b9`
- Normalizer: `amc24_text_normalizer_v1`

## Method Results

| Method | Completed | Exact correct | Exact accuracy | Audited lower-bound correct | Audited lower-bound accuracy | Majority exact | Parser failures | Length stops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 135/135 | 80/135 | 59.26% | 81/135 | 60.00% | 26/45 (57.78%) | 1 | 2 |
| KIVI | 135/135 | 58/135 | 42.96% | 58/135 | 42.96% | 20/45 (44.44%) | 1 | 9 |
| PatternKV | 135/135 | 61/135 | 45.19% | 61/135 | 45.19% | 22/45 (48.89%) | 2 | 7 |
| CAUSAL-V4@25% | 135/135 | 90/135 | 66.67% | 90/135 | 66.67% | 30/45 (66.67%) | 0 | 1 |

## Exact Deltas

| Comparison | Response accuracy delta | Majority accuracy delta |
|---|---:|---:|
| CAUSAL-V4@25% - FP16 | 7.41 pp | 8.89 pp |
| CAUSAL-V4@25% - KIVI | 23.70 pp | 22.22 pp |
| CAUSAL-V4@25% - PatternKV | 21.48 pp | 17.78 pp |

## Scoring Caveat

- Exact normalized-string scoring counts FP16 `12B_23:r0` as incorrect even though `\frac{\sqrt{2}+1}{2}` and `\frac{1+\sqrt{2}}{2}` are mathematically equivalent.
- Therefore FP16 exact response score `80/135` should be read as at least `81/135` under audited equivalence. Majority scores in this summary remain exact-scored.

## Artifacts

- `final_summary.json`: machine-readable frozen summary.
- `method_summary.csv`: method-level table.
- `per_question_summary.json`: question-level N3 votes and correctness.
- `existing_result_provenance_audit.md`: provenance and scorer caveat audit.
