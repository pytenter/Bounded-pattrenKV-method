# S2B-2 Final Report: Pattern Centroid Cost Decomposition

## Outcome

`RECOMMENDED_NEXT_PHASE=CENTROID_PATH_OPTIMIZATION`

S2B-2 added benchmark-only centroid ablation modes and measured V2/V4 Value-attention lanes at 8K, 16K, and 32K. Production CUDA semantics remain unchanged: the normal Python wrapper still calls the original FULL entry, and debug modes are only reachable through explicit benchmark/test calls.

## Benchmark Method

- GPU: NVIDIA GeForce RTX 3090 on physical GPU5
- CUDA events
- Contexts: 8192, 16384, 32768
- Rounds: 5
- Warmup: 30
- Timed iterations: 200
- All key round-to-round CV values are <= 5%

## FULL Kernel Baseline

| Lane | 8K | 16K | 32K |
|---|---:|---:|---:|
| V2 FULL | 114.688 us | 223.232 us | 441.344 us |
| V4 FULL | 96.256 us | 97.280 us | 161.792 us |

## Residual-Only

| Lane | 8K | 16K | 32K |
|---|---:|---:|---:|
| V2 RESIDUAL_ONLY | 100.352 us | 102.400 us | 194.560 us |
| V4 RESIDUAL_ONLY | 87.040 us | 88.064 us | 97.280 us |

## Estimated Centroid Cost

This is `FULL - RESIDUAL_ONLY`, so it is `NOT_STRICTLY_ADDITIVE` and should be treated as approximate ablation cost.

| Lane | 16K | 32K |
|---|---:|---:|
| V2 centroid | 120.832 us / 54.1% | 246.784 us / 55.9% |
| V4 centroid | 9.216 us / 9.5% | 64.512 us / 39.9% |

## Mask Density

At V2 16K, FULL latency rises from 160.768 us at 0% mask density to 336.896 us at 100% density. That is +109.6%, showing the centroid path scales with active Pattern tokens.

## Assignment Contention

Uniform assignment FULL median is 335.872 us. Skewed assignment FULL median is 598.016 us, a 1.780x slowdown. This supports atomic contention as a structural centroid-path bottleneck.

## GQA Audit

GQA centroid reuse opportunity is `YES`: centroid table, mask, assignment, and compressed V metadata are repeated across four query heads per KV head. Query-head attention weights remain unique, so reuse needs a later kernel redesign.

## Correctness

- FULL max_abs_error: 3.0517578125e-05
- FULL relative_L2: 0.0004357079742476344
- FULL cosine min: 0.9999998807907104
- NaN: 0
- Inf: 0
- Production FULL unchanged vs debug FULL: YES
- Generation smoke: PASS, 128 generated tokens

## Decision

`CENTROID_PATH_OPTIMIZATION`

Reason: V2 centroid overhead is about 55.9% at 32K, mask density strongly affects latency, and skewed assignment creates a large slowdown. The next phase should optimize the centroid histogram/atomic/table contribution path before attempting broader GQA-aware redesign or fixed-page ABI work.

## Validation

- compileall: PASS
- pytest: 473 passed, 1 warning
- git diff --check: PASS
- Full AIME24: NO
- AIME25: NO
- GPQA: NO
- vLLM: NO
- SGLang: NO
