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

## Matched-B grid
```bash
CUDA_VISIBLE_DEVICES=5 \
PATTERNKV_FIXED_SPLIT_SOFTMAX=1 \
PATTERNKV_ACTIVE_BATCH_CACHE=1 \
PATTERNKV_SYSTEM_PROFILE=0 \
/data/zypan/.local/share/mamba/envs/patternkv/bin/python <inline matched-B runner>
```

Matched-B used context=2048, decode=8, B=1/2/4/8, total_requests=max(2*B, B+1), one warmup and three measured runs for valid points.

## Capacity grid
```bash
CUDA_VISIBLE_DEVICES=5 \
PATTERNKV_FIXED_SPLIT_SOFTMAX=1 \
PATTERNKV_ACTIVE_BATCH_CACHE=1 \
PATTERNKV_SYSTEM_PROFILE=0 \
/data/zypan/.local/share/mamba/envs/patternkv/bin/python <inline capacity runner>
```

Capacity used context=4096, decode=8, powers-of-two B until first OOM per method.
