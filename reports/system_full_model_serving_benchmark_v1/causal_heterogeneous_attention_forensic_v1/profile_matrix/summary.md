# Full-Model Post-Scaling Bottleneck Forensic V1

## Memory Finding

C4096 B4 OOM occurs during initial full-batch prefill for both FP16 and CAUSAL. Decode-only timing remains uncontaminated.
Both OOM traces fail at the full-vocabulary `logits.float()` allocation during prefill, requesting 7.83 GiB after the model and prefill activations/workspace have already consumed most of the 24GB device.
The dominant full-lifecycle peak is model weights plus prefill activation/logit/workspace pressure, not persistent historical KV payload.

## OOM Points

## Persistent Cache Breakdown

## Quantitative Memory Accounting

- Model parameters/buffers are ~16.061 GB for both FP16 and CAUSAL.
- FP16 C4096 B2 persistent KV is ~1.082 GB, estimated from decode-ready lifecycle delta and matching the theoretical 32 layers x B2 x K/V x 8 KV heads x 4096 x 128 x FP16 layout.
- CAUSAL C4096 B2 persistent decode-ready cache tensors total ~0.299 GB: compressed K 0.063 GB, V2 payload 0.047 GB, V4 payload 0.031 GB, FP16 sink/recent/pending tail 0.067 GB, centroid/metadata/scale/zero about 0.090 GB.
- FP16 C4096 B2 prefill peak is 23.514 GB, leaving ~1.22 GB free; CAUSAL C4096 B2 prefill peak is 22.748 GB, leaving ~1.62 GB free.
- The persistent-cache advantage at C4096 B2 is roughly 0.78 GB, while the B4 prefill failure asks for another 7.83 GiB allocation. That gap explains why compressed historical KV does not move max B from 2 to 4.
- CAUSAL has no evidence of persistent duplicate historical FP16 K/V cache: repaired structural counters remain historical FP16 K/V materialization = 0, and tensor ownership has no FP16 historical cache category for CAUSAL.

## Decode Finding

Formal repaired C2048 B1 TPOT is ~28.0 ms/token for FP16 and ~187.9 ms/token for CAUSAL, so the real incremental latency is ~160 ms/token.
The profiler has overhead, so absolute profiled TPOT is not used as the formal metric. Component percentages are used to attribute where the CAUSAL path spends time.

Top CAUSAL C2048 B1 profiled decode components:

- `decode_layer_self_attention`: ~60% of profiled decode time.
- `page_batch_pack`: ~46% of profiled decode time, nested inside cache append/flush work.
- `decode_layer_post_attention_rmsnorm`: ~15%.
- `decode_layer_mlp`: ~13%.
- `value_fp16_tail`: ~16%, nested inside attention.
- `attention_softmax` and `cache_append`: each ~7%.
- `qk_quantized_history`: ~1-2%, so compressed historical QK alone is not the dominant current bottleneck.

Because these ranges are nested, exact additive closure is not valid. The evidence supports a multi-component root cause, dominated by per-layer CAUSAL attention/cache-update/page-pack/tail work plus nontrivial RMSNorm/MLP full-model tax.

## Scaling Diagnosis

Context scaling does not improve the CAUSAL/FP16 ratio because the measured dominant CAUSAL costs are not just historical-QK memory traffic that grows with context. A large fraction is per-token/per-layer fixed or tail/cache-update/page-pack work.
B scaling does not improve the ratio because FP16 also scales strongly with B on the same model path, while CAUSAL carries per-layer compressed-cache mutation and page-pack work that scales with batch/output tokens rather than being amortized away.

## Optimization Priorities

- P0 memory: remove or avoid the prefill full-vocab logits.float peak for full-lifecycle capacity. Expected effect: capacity; low direct decode TPOT impact.
- P1 decode: attack CAUSAL cache append/flush/page-pack/value-tail path. Expected effect: TPOT and throughput; moderate semantic risk because it touches production cache update/value path.
- P2 decode: reduce fixed-split softmax/RMSNorm/runtime overhead after P1. Expected effect: incremental TPOT; lower capacity impact.

## Classification

- TASK_CLASSIFICATION: FULL_MODEL_POST_SCALING_BOTTLENECK_FORENSIC_V1_SUPPORTED
- MEMORY_ROOT_CAUSE: PREFILL_NON_KV_PEAK_DOMINATED
- DECODE_ROOT_CAUSE: MULTI_COMPONENT
