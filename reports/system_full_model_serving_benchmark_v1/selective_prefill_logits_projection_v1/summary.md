# SELECTIVE_PREFILL_LOGITS_PROJECTION_V1

## Status

`SELECTIVE_PREFILL_LOGITS_PROJECTION_V1_SUPPORTED`. Generation prefill now selects final valid hidden rows before `lm_head`, so normal next-token prefill projects `[B,H] -> [B,V]` instead of `[B,L,H] -> [B,L,V]`.

## Exact Old Failure

Before P0, both FP16 and CAUSAL C4096 B4 failed in `initial_prefill` at full-vocab `logits.float()`. The failed allocation was 7.83 GiB because the path materialized full-sequence full-vocab FP32 logits before taking last-token logits.

## Select-Before-LM-Head Design

The repaired path runs the decoder body with cache construction, selects each request's final valid hidden row using request-local valid length or attention mask semantics, then applies `lm_head` only to the selected rows. Decode remains unchanged. `PATTERNKV_SELECTIVE_PREFILL_LOGITS=0` preserves an explicit fallback to the old full-logits path for debug.

## Correctness

Actual-model oracle checks passed for FP16 and CAUSAL at B1/B2 C32. The maximum observed absolute difference versus full-logits reference was 0.007812, maximum relative L2 was 5.030920e-05, and top1 matched for every checked row. CPU/mock tests cover B1, ragged B2, ragged B4, full-logits equality, and proof that `lm_head` input rows are pruned before projection.

## LM Head Shapes

- B1 C32: hidden `[1, 32, 4096]`, lm_head input `[1, 4096]`, logits `[1, 128256]`, rows 32 -> 1.
- B2 C32: hidden `[2, 32, 4096]`, lm_head input `[2, 4096]`, logits `[2, 128256]`, rows 64 -> 2.
- Expected C4096 B4 generation prefill row reduction: 16384 hidden rows -> 4 hidden rows before vocab projection.

## Memory Before / After

- FP16 C4096 B2 prefill peak allocated: 23.514 GB -> 18.118 GB.
- CAUSAL C4096 B2 prefill peak allocated: 22.748 GB -> 17.685 GB.
- FP16 C4096 B4: old OOM -> PASS; after-P0 prefill peak allocated 20.165 GB, decode peak allocated 22.553 GB.
- CAUSAL C4096 B4: old OOM -> PASS; after-P0 prefill peak allocated 19.268 GB, decode peak allocated 16.959 GB.
- The old B4 7.83 GiB logits allocation is absent after P0; B4 has no OOM in after-P0 memory forensic.

## Capacity After P0

- FP16 C4096: max PASS B4, first OOM B8, own-max decode throughput 98.89 tok/s.
- CAUSAL C4096: max PASS B8, first OOM B16, own-max decode throughput 26.69 tok/s.
- Capacity ratio: 2.0x.
- Own-max decode throughput ratio CAUSAL/FP16: 0.270x.

## Decode Regression Guard

C2048 matched-B after P0 remains in the repaired decode-only range: FP16 B1 28.75 ms/token, CAUSAL B1 195.65 ms/token. B1-B8 protocol points all passed.

## Protocol Invariants

Timed-window prefill calls, prefill tokens, refill calls, and membership changes are all zero in the repaired after-P0 run. Serial request, attention, MLP, and RMSNorm dispatch counters are zero; fallback count is zero.

## New OOM Boundary

After P0, the old logits materialization blocker is gone. The remaining capacity boundary is FP16 B8 near device exhaustion on a 66 MiB allocation, and CAUSAL B16 on a 1.75 GiB allocation. This supports moving from `CHUNKED_PREFILL_V1` only if the next priority is more capacity; for throughput, the next priority remains decode/runtime optimization such as `DIRECT_COMPRESSED_PAGE_APPEND_V1`.

## Final Classification

- `SELECTIVE_PREFILL_LOGITS_PROJECTION_V1_SUPPORTED`
- `POST_P0_FULL_MODEL_CONCURRENCY_ADVANTAGE = SUPPORTED`
- `POST_P0_FULL_MODEL_MEMORY_TO_DECODE_THROUGHPUT_TRANSLATION = NOT_SUPPORTED`
- `COMMIT_CREATED = false`
- `PUSHED = false`
