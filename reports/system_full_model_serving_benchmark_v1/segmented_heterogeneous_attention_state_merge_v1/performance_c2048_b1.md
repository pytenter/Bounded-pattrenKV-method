# C2048 B1 Performance

## Formal Decode-Only Worker

- GPU: physical GPU1 via `CUDA_VISIBLE_DEVICES=1`.
- Workload: DeepSeek-R1-Distill-Llama-8B, C2048, B1, decode=8.
- Old path env: `PATTERNKV_SEGMENTED_STATE_MERGE=0`.
- New path env: `PATTERNKV_SEGMENTED_STATE_MERGE=1`.
- Protocol invariants: prefill calls/tokens/refills/membership changes in timed window all zero.

| Path | TPOT ms/token | tok/s | status |
| --- | ---: | ---: | --- |
| old CAUSAL | 191.697 | 5.216 | PASS |
| new CAUSAL repeat best | 281.952 | 3.547 | PASS |
| new CAUSAL other run | 384.408 | 2.601 | PASS |

Best observed new-path speedup vs old: `0.680x`.

Best observed absolute ms/token saved: `-90.255 ms/token`.

The primary performance gate is not supported. No B scaling or context scaling was run.

