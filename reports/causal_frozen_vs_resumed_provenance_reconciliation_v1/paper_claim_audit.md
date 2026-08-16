# Paper Claim Audit

| Claim | Status | Rationale |
|---|---|---|
| CAUSAL full-model C2048/B1 ~155 ms/token | PRIMARY_SUPPORTED | Frozen stored row is 154.671 ms/token; 8d/50a canonical reruns are ~165-167 ms/token on GPU 1, same regime. |
| CAUSAL full-model capacity 2x FP16 | PRIMARY_SUPPORTED_WITH_ALLOCATOR_PROTOCOL | Frozen capacity B8 vs FP16 B4 is preserved when `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is part of the formal protocol. |
| CAUSAL current same-harness C2048/B1 ~275 ms/token | SUPERSEDED | Stored resumed row did not reproduce with the same current worker on the same GPU. |
| CAUSAL current same-harness capacity = FP16 | SUPERSEDED | Current B8 OOM reproduces only when the freeze allocator env is omitted; with the frozen env, B8 passes. |
