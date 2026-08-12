# S2B-2A Final Report: Centroid Histogram / Atomic Optimization

## Outcome

`CENTROID_HISTOGRAM_OPTIMIZATION_SUPPORTED`

Candidate B, private per-warp shared-memory histograms, is retained in production. It changes only the centroid histogram accumulation strategy inside the fused Value-attention kernel. Selector, cache layout, quantization, residual path math, Pattern assignments, masks, and centroid values are unchanged.

## Baseline Contention

- V2 normal @16K: 226.30399465560913 us
- V2 normal @32K: 445.43999433517456 us
- V2 skewed @16K: 612.3520135879517 us
- V2 skewed @32K: 1212.4160528182983 us
- Skew slowdown @32K: 1.808x
- Contention reproduced: YES

## Candidate A

Warp-local aggregation before atomic reduced logical atomics but was much slower due to warp matching/shuffle overhead. It is rejected.

## Candidate B

Candidate B uses `s_Sacc_private[4][Mcent]`; each warp builds its own histogram row for one quarter of the 128-token tile, and centroid-table contribution sums the four rows.

- Normal V2 @16K: 161.79199516773224 us, speedup 1.399x
- Normal V2 @32K: 309.2480003833771 us, speedup 1.440x
- Skewed V2 @32K: 551.9359707832336 us, speedup 2.197x
- Estimated normal V2 @32K logical atomic reduction: 33.5%

## Mixed-V And E2E

Fair debug FULL vs per-warp mixed-V @32K: 1014.7839784622192 us -> 827.3919820785522 us, speedup 1.226x.

Production optimized E2E @32K median TPOT: 126.4634895324707 ms -> 121.64556884765625 ms, speedup 1.040x.

## V4 Guard

V4 @32K: 163.83999586105347 us -> 112.64000087976456 us. No V4 regression.

## Correctness

Baseline-vs-candidate correctness passed for normal, uniform, skewed, and all-same-centroid cases. Max baseline-vs-candidate abs error in recorded rows is 0.0, cosine >= 0.9999998807907104.

## Build

- Extension rebuilt: YES
- Loaded binary: /data/zypan/Bounded-pattrenKV-pseudodecode-3090/quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so
- SHA256: a4fe6f34daad5e05704f197fbe198e4a66df7e964f113627f2242d6c41f0d9b1

## Next Task

`CENTROID_TABLE_CONTRIBUTION_OPTIMIZATION`

The histogram bottleneck is reduced. The next profile target should be the remaining centroid-table contribution (`Sacc[c] * centroid[c]`) and/or a follow-up GQA-aware kernel redesign depending on the post-optimization profile.

## Validation

- compileall: PASS
- pytest: 481 passed, 1 warning
- git diff --check: PASS
- Full AIME24: NO
- AIME25: NO
- GPQA: NO
- vLLM: NO
- SGLang: NO
