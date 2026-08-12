# CAUSAL-V4@25% — Frozen Algorithm Checkpoint v1

Frozen algorithm commit: `c73aeed3247c136859f695d5b238eeb357434b17`

Dataset: `AIME24`

Model: `DeepSeek-R1-Distill-Llama-8B`

Formal generations: `360 / 360`

Classification: `SUPPORTED`

## Frozen Result

| Method | Correct / 90 | Accuracy |
|---|---:|---:|
| FP16 | 45/90 | 50.00% |
| PATTERN_BASE | 32/90 | 35.56% |
| RANDOM_V4_25 | 36/90 | 40.00% |
| CAUSAL_V4_25 | 45/90 | 50.00% |

CAUSAL_V4_25 improves over PATTERN_BASE by +14.44 pp.

CAUSAL_V4_25 improves over the storage-matched RANDOM_V4_25 control by +10.00 pp.

CAUSAL_V4_25 matches FP16 aggregate AIME24 accuracy in this specific evaluation.

The CAUSAL - RANDOM paired question-cluster bootstrap 95% CI crosses zero, so this release does not claim statistical significance over RANDOM at the 95% level.

## Frozen Algorithm

- Method: PatternKV
- K bits: 2
- Base V bits: 2
- Selected V bits: 4
- V4 budget: 25%
- Sink: 16 tokens
- Recent: 128 tokens
- Residual: 128 tokens
- Group size: 128
- Selector: causal_v4
- Random selector seed: 20260809
- Full 128-token window: 32 V4 tokens and 96 V2 tokens

These semantics are frozen for v1. Later systems work must not change the budget, selector definition, importance score, local V2-to-V4 gain definition, Sink16, Recent128, Residual128, or group_size=128.

## Bit Accounting

The formal payload-and-metadata budget is:

| Method | Formal budget |
|---|---:|
| PATTERN_BASE | 2.25 bit/element |
| RANDOM_V4_25 | 2.50048828125 bit/element |
| CAUSAL_V4_25 | 2.50048828125 bit/element |

`2.500488 bit/element` is not the complete physical KV-cache memory footprint. Physical accounting for sink storage, recent storage, centroid storage, assignment tensors, dtype overhead, and allocator/alignment overhead is intentionally deferred to the systems branch.

## Authoritative Artifacts

The formal result artifacts are under `reports/aime24_full_causal25_quality_4gpu/`. This release directory records metadata and checksums for freezing and handoff only; it does not modify the algorithm or completed AIME24 results.
