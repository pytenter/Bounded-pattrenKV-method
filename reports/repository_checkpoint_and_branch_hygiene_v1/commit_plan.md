# Semantic Commit Plan

The dirty tree contains multiple completed scientific phases. The proposed split is the smallest safe set that keeps direct production changes with their regression tests and evidence without using a single giant commit.

1. `fix: preserve request-local fixed-split semantics`
   - Scope: request-invariant QK/softmax/value, ragged valid lengths, fixed-split CUDA softmax, fused page seq-len handling, batch-invariant decode/RMSNorm support, direct tests and core ragged reports.

2. `feat: support request lifecycle continuous batching`
   - Scope: request lifecycle manager, dynamic add/remove, continuous batching tests, lifecycle reports.

3. `fix: repair full-model decode-only benchmark lifecycle`
   - Scope: full-model serving benchmark harness, decode-only protocol repair scripts/tests/reports, serving benchmark v1 evidence.

4. `feat: select final prefill rows before lm head projection`
   - Scope: selective prefill logits tests and reports, capacity after-P0 evidence.

5. `test: close post-scaling full-model capacity forensic`
   - Scope: post-scaling bottleneck forensic script/report, capacity/memory reports.

6. `test: profile heterogeneous causal attention path`
   - Scope: corrected decode profiler lifecycle attribution hooks, heterogeneous attention profile subranges, forensic reports.

7. `docs: document repository checkpoint and branch map`
   - Scope: repository hygiene reports and `docs/BRANCH_MAP.md` after branch inventory.

Files marked `RAW_EXPERIMENT_ARTIFACT`, `GENERATED_LOG`, or `review` in the manifest should not be staged automatically. Binary `forensics/*.pt`, probe directories, smoke raw runs, and logs require manual review.
