# B4 Request-Count Kernel Geometry Fix

## FIRST_BAD_ATTENTION_SUBCOMPONENT

Pre-fix frozen evidence localized the request-B divergence for `[A,B]` versus request count `>=3` to `ATTENTION_PRE_O_PROJ` at decode step 1, layer 0. Within attention, the first bad subcomponent was `masked_scores` after exact request-local Q/K/V, exact valid request-local cache slices, exact packed-K CUDA scores, and exact reference raw QK for the request-local slice.

Post-fix validation:

- `scripts/b4_attention_microtrace.py --good A,B --bad A,B,C,D`
- `first_bad_attention_subcomponent: null`
- request-local Q/K/V, raw QK scores, scaled scores, masked scores, softmax probabilities, probability-times-V, packed Value kernel output, and merged-head pre-O projection all compare exact for tracked request B.

## EXACT_ROOT_CAUSE

The full-precision sink, pending, and recent K score regions used batched `torch.matmul` in the segmented attention path. Adding peer requests changed the batched matmul geometry for request B even though request-local Q/K/V/lengths were semantically identical. That changed floating-point accumulation order and produced tiny FP16 score differences that first appeared in the concatenated/scaled/masked score tensor and then propagated into softmax, pre-O output, logits, and persistent state.

This was a numerical accumulation-order dependency caused by request-count-dependent kernel geometry, not evidence of wrong request indexing, peer-request contamination, invalid K/V ownership, mask contamination, or corrupted cache assembly.

## B2_VS_B3_KERNEL_GEOMETRY_DIFFERENCE

The post-fix microtrace preserved the geometry distinction while removing request-B semantic drift:

- Good `[A,B]`: logical batch 2, attention width 256, grid blocks `[64,128,1]`, pages per request 2, seq lens `[128,256]`, v2 tokens 288, v4 tokens 96.
- Bad-trigger geometry `[A,B,C,D]`: logical batch 4, attention width 512, grid blocks `[128,128,1]`, pages per request 4, seq lens `[128,256,384,512]`, v2 tokens 960, v4 tokens 320.
- The packed-K CUDA score path and packed Value fused page path remained exact. The divergent region was the full-precision QK score path for sink/pending/recent segments.

## FILES_CHANGED

- `models/llama_patternkv.py`: added request-invariant full-precision QK score helper and used it for sink, pending, and recent attention score segments.
- `models/segmented_cache.py`: reduced default V-candidate selector chunk size from 128 to 16 to keep the B4 final gate within memory without changing token-local candidate semantics.
- `bench/run_ragged_multistep_correctness.py`: builds independent B1 reference trajectories before ragged replay so the final gate is not polluted by reference interleaving.
- `scripts/b4_attention_microtrace.py`: added/kept attention subcomponent forensic trace for request B step 1 layer 0.
- `tests/test_ragged_k_valid_lengths.py`: added request-invariant QK regression coverage and fixed the helper contract expectation to the explicit fixed-order reference.
- `tests/test_value_direction_screen.py`: added V-candidate block-size equivalence coverage.

## PRODUCTION_FIX

`patternkv_request_invariant_qk_scores(query_states, key_states, num_key_value_groups)` computes full-precision segment scores through request-local fixed-order elementwise multiply plus `sum(dim=-1)`, after GQA `repeat_kv`. The segmented attention path now uses this helper for `sink_k`, `pending_k`, and `recent_k` instead of batched `torch.matmul`.

The fix keeps true batched execution and compressed-domain runtime behavior. It does not serialize request forwards, does not add serial attention dispatches, does not special-case B=3, and does not loosen correctness tolerances.

The V-candidate chunk-size change is a memory-safety production fix for final-gate B4 flush validation. Token candidate selection is independent by token, and unit coverage confirms block-size equivalence.

## REQUEST_COUNT_REGRESSION_MATRIX

- `[A,B]`: PASS through B2 `[384,513]` 16-step gate.
- B2 reorder `[A,B]` versus `[B,A]`: PASS.
- `[A,B,C]`: PASS via B4 request-count regression pytest and post-fix no-first-bad attention trace behavior.
- `[A,B,D]`: PASS via B4 request-count regression pytest.
- `[A,B,C,D]`: PASS through B4 `[384,513,642,771]` 16-step gate.
- Independent flush: PASS.
- Observed B4 flush steps: `{"D": 13, "C": 14, "B": 15, "A": 16}`.
- Persistent state ownership: no cross-request centroid update detected, no cross-request page contamination detected.

## ORIGINAL_B2_384_513_RESULT

The historical `B2 [384,513] relL2 ~= 0.0271493` was pre-fix evidence only. Current acceptance result is:

- `b2_16step_pass: true`
- `b2_reorder_16step_pass: true`
- `max_logit_relative_l2: 0.0025098149199038744`
- max request/step: `A`, step `16`

## SYSTEM_INVARIANTS

- `SERIAL_REQUEST_FORWARD_DISPATCHES = 0`
- `SERIAL_ATTENTION_DISPATCHES = 0`
- `SERIAL_VALUE_REDUCTION_REQUEST_LOOPS = 0`
- `HISTORICAL_FP16_K_MATERIALIZATION = 0`
- `HISTORICAL_FP16_V_MATERIALIZATION = 0`
- `FALLBACK_COUNT = 0`
- `TRUE_BATCH_PRESERVED = true`
- `COMPRESSED_DOMAIN_RUNTIME_PRESERVED = true`

Final gate counters:

- `serial_request_dispatches: 0`
- `bi_kproj_serial_request_dispatches: 0`
- `historical_fp16_k_materialization: 0`
- `historical_fp16_v_materialization: 0`
- `fallback_calls: 0`

## TARGETED_PYTEST

Command:

```bash
/data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python -m pytest tests/test_ragged_k_valid_lengths.py tests/test_value_direction_screen.py tests/test_b4_request_count_ragged_divergence.py tests/test_fused_page_batch_operator.py
```

Result: `81 passed in 6.97s`.

## FULL_PYTEST

Command:

```bash
/data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python -m pytest
```

Result: `942 passed in 26.11s`.

## COMPILEALL

Command:

```bash
/data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python -m compileall models/llama_patternkv.py models/segmented_cache.py bench/run_ragged_multistep_correctness.py scripts/b4_attention_microtrace.py tests/test_ragged_k_valid_lengths.py tests/test_value_direction_screen.py
```

Result: PASS.

## FINAL_CLASSIFICATION

`PATTERNKV_RAGGED_MULTI_STEP_CORRECTNESS_SUPPORTED`

Next accepted task: `IMPLEMENT_DYNAMIC_REQUEST_ADD_REMOVE_MVP`.
