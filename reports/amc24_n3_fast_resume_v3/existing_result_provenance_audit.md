# AMC24 N3 Fast Resume Provenance Audit

- Frozen response set: `[0, 1, 2]`
- Frozen seeds: `[42, 43, 44]`
- PatternKV: `SCIENTIFIC_MATCH`
- CAUSAL: `SCIENTIFIC_MATCH`

Existing formal results are preserved and reused whenever the task is completed.

## Scoring Caveat

- The current scoring path uses exact normalized-string matching rather than symbolic equivalence.
- Audited false negative: FP16 `12B_23:r0` predicts `\frac{\sqrt{2}+1}{2}`, while the gold key is `\frac{1+\sqrt{2}}{2}`. This is mathematically equivalent but counted incorrect.
- Treat the raw FP16 N3 single-sample score `80/135` as at least `81/135` under this audited equivalence; FP16 majority remains `26/45`.
- Treat the CAUSAL-over-FP16 margin as an observed N3 result with scorer-normalizer limitations, not a robust superiority claim.
