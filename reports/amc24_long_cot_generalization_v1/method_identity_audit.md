# Method Identity Audit

## Planned AMC24 Method Configurations

| Method | Required Runtime Identity | Status |
| --- | --- | --- |
| FP16 | no quantized KV path | NOT_EXECUTED |
| KIVI | `kivi_paper_g128`, K/V INT2, group 128 | NOT_EXECUTED |
| PatternKV | `patternkv_paper`, K/V INT2, group 128, 32 base patterns | NOT_EXECUTED |
| CAUSAL-V4@25% | PatternKV segmented rolling, K INT2, V2 base with top 25% eligible V INT4, sink 16, recent/residual 128, group 128, causal selector | NOT_EXECUTED |

## Frozen CAUSAL Semantics

The required frozen algorithm checkpoint is `c73aeed3247c136859f695d5b238eeb357434b17`. No implementation or runtime change was made by this task.

## AIME24 Pattern Base vs Paper PatternKV

`PATTERN_BASE_AIME24_IS_PAPER_PATTERNKV_BASELINE = uncertain`.

The historical AIME24 Pattern Base configuration is `pattern_rolling_k2v2_s16_r128` with `base_v2`, K/V INT2, group 128, sink 16, recent 128, and residual 128. The standard `patternkv_paper` configuration matches the core K/V INT2, group-128, 32-pattern architecture, but the standard baseline metadata inspected for LongBench does not preserve all sink/recent/runtime-path fields. This naming relation must not be assumed to establish equivalence in an AMC24 table.
