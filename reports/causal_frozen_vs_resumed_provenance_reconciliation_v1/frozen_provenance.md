# Frozen Provenance

Frozen report path: `reports/system_full_model_serving_benchmark_v1/final_serving_benchmark_freeze_v1/`.

Stored environment reports source SHA `c59a88cce9a967a810f1de7601a3c531f1f29bc0`, branch `sys/causal-v4-25-kernel-v1`, Python `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`, torch `2.4.1+cu124`, CUDA runtime `12.4`, and physical GPU 2 UUID `GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca`. The frozen commit `8d60485` preserves these reports and the freeze script.

The freeze runner `scripts/final_serving_benchmark_freeze_v1.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `PATTERNKV_FP16_TAIL_VALUE_FUSION=1`, `PATTERNKV_FIXED_SPLIT_SOFTMAX=1`, `PATTERNKV_SELECTIVE_PREFILL_LOGITS=1`, `PATTERNKV_ACTIVE_BATCH_CACHE=1`, and `PATTERNKV_SYSTEM_PROFILE=0` for formal point subprocesses. It uses `bench.full_model_serving_benchmark.load_causal_model`, `PatternKVAdapter`, one subprocess per point, prefill before timing, zero timed prefill/refill/membership changes, and zero fallback.

Stored frozen C2048/C4096 D8 rows and the same-GPU reproduction both show the fast regime. The stored long-decode file is C2048/B1/D256, not C4096/B1/D256.
