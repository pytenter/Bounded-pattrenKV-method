# Production Fix

Production files changed: `models/llama_patternkv.py`, `quant/batch_invariant_kproj.py`, `tests/test_bi_kproj_prefill_runtime.py`, `bench/run_actual_model_bi_prefill_runtime.py`, `bench/run_bi_vproj_cost_benefit.py`, `bench/run_prefill_projection_mode_policy.py`, `bench/run_ragged_multistep_correctness.py`.

K decode now reuses existing BI KProj V2 instead of `self.k_proj` whenever projection mode is `bi_k` or `bi_kv`. A secondary `recent_v` divergence was exposed, so strict `bi_kv` was also extended to route decode V through the same existing BI linear projection. No algorithm bits, selector, cache layout, centroid semantics, or quantization parameters were changed.
