# Commit Diff Audit

Relevant changes from `8d60485` to `50a9a27`:

| File | Classification | CAUSAL reachable? | Evidence |
|---|---|---|---|
| `bench/paper_baseline_system_comparison.py` | BENCHMARK_PROTOCOL_REACHABLE | Yes | New four-method paper wrapper; omits the frozen runner's `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` default. |
| `bench/full_model_serving_benchmark.py` | BENCHMARK_PROTOCOL_REACHABLE / BASELINE_ONLY | Conditional | Adds KIVI/PatternKV-paper adapters and loader plumbing. CAUSAL still maps to `PatternKVAdapter`; timing loop is materially unchanged for CAUSAL. |
| `models/llama_patternkv.py` | PATTERNKV_PAPER_ONLY / SHARED_RUNTIME | No for CAUSAL mixed-V path | Adds base-V2 page Value path when `operator_ready_page_pools` exist and Value precision is not mixed. CAUSAL V4 uses mixed precision and continues through `patternkv_mixed_value_attention`. |
| `models/segmented_cache.py` | PATTERNKV_PAPER_ONLY | No for CAUSAL mixed-V path | Adds base-V2 operator-ready page-pool construction; guarded by `normalize_value_precision_selector(...) == base_v2`. CAUSAL V4 selector is mixed, so this branch is not entered. |
| `quant/page_batch.py` | PATTERNKV_PAPER_ONLY / SHARED_RUNTIME | Conditional but not performance-regression producing | Adds batched centroid indexing for 4D centroid tensors. Direct 50a A/B matches 8d, rejecting this as a CAUSAL slowdown cause. |

Conclusion: no production diff explains a CAUSAL slowdown under matched env. The only reproduced current/frozen conflict is benchmark memory lifecycle drift from the allocator env omission in the current paper wrapper.
