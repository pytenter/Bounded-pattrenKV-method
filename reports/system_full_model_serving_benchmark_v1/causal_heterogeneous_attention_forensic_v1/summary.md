# CAUSAL Heterogeneous Attention Forensic V1

## Status

`CAUSAL_HETEROGENEOUS_ATTENTION_FORENSIC_V1_SUPPORTED`: repository checkpoint recovered the project `patternkv` environment and validated the dirty tree with targeted pytest and full pytest. This run did not optimize kernels or change frozen CAUSAL-V4@25% semantics.

## Headline Finding

`decode_layer_self_attention` is the largest CAUSAL bucket because the current runtime integrates heterogeneous storage by materializing concatenated score/probability tensors and launching separate segment Value paths. At C2048 B1 under profiler, CAUSAL attention is 1320.62 ms/run versus FP16 attention 251.39 ms/run, a delta of 1069.23 ms/run or 133.65 ms/output-token.

The largest measured CAUSAL-specific subrange at C2048 B1 is `value_fp16_tail`: 337.29 ms/run, 42.16 ms/output-token, 16.24% of profiled decode wall time. It is broad but accurate: it contains FP16 Value for sink, pending, and recent; it does not include historical V2/V4. Its subranges are sink 114.43 ms, pending 96.59 ms, recent 102.35 ms per run.

## Component Attribution

C2048 B1 CAUSAL, decode=8, decode-only counters reset after prefill:

- `decode_layer_self_attention`: 1329.88 ms/run, 64.03% profiled decode.
- `value_fp16_tail`: 337.29 ms/run, 42.16 ms/token.
- `cache_append`: 154.04 ms/run, 19.25 ms/token.
- `attention_softmax`: 147.73 ms/run, 18.47 ms/token.
- `mixed_v_page_pool_operator`: 48.69 ms/run, 6.09 ms/token.
- `qk_int2_history`: 42.07 ms/run, 5.26 ms/token.
- `page_batch_pack`: 0 in decode=8; the corrected profile no longer supports direct page append as current TPOT priority.

## Dequantization

Historical V2/V4 is `PARTIALLY_FUSED`: the page-pool CUDA kernel reads packed low-bit payload, unpacks/dequantizes in the kernel, applies centroid correction when needed, and accumulates directly into the output. No full historical FP16 V global tensor was found. The path still has global intermediates for scores/probabilities and wrapper dtype/layout normalization.

## Merge Architecture

Current history/tail merge is not attention-state merge. It concatenates segment logits, computes one global softmax, then computes packed-history and FP16-tail Value outputs using normalized probabilities and sums output tensors. The fixed-split softmax kernel internally has online `max` and `sum` state, so FlashInfer-style exact state merge is `SUPPORTED_WITH_MODERATE_ABI_CHANGE`, not naturally exposed today.

## Scaling Diagnosis

Context 2K to 8K does not improve CAUSAL/FP16 ratio because a large part of CAUSAL overhead is fixed per layer/token or tied to FP16 tail, softmax, and cache append rather than only historical low-bit traffic. Historical mixed V grows at C8192 (20.00 ms/token), but it is still one component in a multi-component path.

B1 to B4 does not substantially improve ratio because segment ranges remain per-layer and the runtime still executes separate QK/softmax/Value/cache paths. B4 increases mixed V and QK history work in iteration time rather than amortizing the heterogeneous integration overhead away.

## Boundary Probe

Long decode C2048 B1 decode=136 shows boundary work starts after pending reaches 128: `page_batch_pack` 55.60 ms over 32 calls and `pack_window` 695.67 ms over 32 calls. Amortized this is 0.41 ms/token for page pack and 5.12 ms/token for pack window. Boundary maintenance is LOW for decode=8 and MEDIUM for long decode >=128.

## Root Cause

`HETEROGENEOUS_ATTENTION_ROOT_CAUSE = MULTI_COMPONENT`. The top next target is not direct page append. The best supported next optimization is `SEGMENTED_HETEROGENEOUS_ATTENTION_STATE_MERGE_V1`, because it attacks the score/probability materialization and separate segment integration that make FP16 tail, softmax, and Value paths expensive together. A narrow FP16-tail-only optimization has an Amdahl upper bound of about 1.19x on the profiled C2048 B1 run if made free, so it is useful but not sufficient alone.
