# Exact Commands

## Preflight
```bash
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat
git diff --check
nvidia-smi
```

## Context grid
```bash
CUDA_VISIBLE_DEVICES=5 \
PATTERNKV_FIXED_SPLIT_SOFTMAX=1 \
PATTERNKV_ACTIVE_BATCH_CACHE=1 \
PATTERNKV_SYSTEM_PROFILE=0 \
/data/zypan/.local/share/mamba/envs/patternkv/bin/python <inline context-scaling runner>
```

The runner used `BenchmarkConfig(method, context, decode=8, active_capacity=1, total_requests=2)` for contexts `256,2048,4096,8192`, methods `FP16_FULL_MODEL` and `CAUSAL_V4_25_FULL_MODEL`, one warmup and three measured runs for valid points. CAUSAL structural checks used `PATTERNKV_SYSTEM_PROFILE=1` after headline measurements and were not used for throughput.
