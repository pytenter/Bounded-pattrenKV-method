# Exact Commands

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat
git diff --check
nvidia-smi
```

```bash
CUDA_VISIBLE_DEVICES=5 \
PATTERNKV_FIXED_SPLIT_SOFTMAX=1 \
PATTERNKV_ACTIVE_BATCH_CACHE=1 \
PATTERNKV_SYSTEM_PROFILE=0 \
/data/zypan/.local/share/mamba/envs/patternkv/bin/python \
  scripts/reconcile_scaling_path_attention_roofline.py --runs 3 --warmup 1
```

```bash
CUDA_VISIBLE_DEVICES=5 \
PATTERNKV_FIXED_SPLIT_SOFTMAX=1 \
PATTERNKV_ACTIVE_BATCH_CACHE=1 \
PATTERNKV_SYSTEM_PROFILE=0 \
/data/zypan/.local/share/mamba/envs/patternkv/bin/python \
  scripts/reconcile_scaling_path_attention_roofline.py --profile-only
```

```bash
CUDA_VISIBLE_DEVICES=5 PATTERNKV_FIXED_SPLIT_SOFTMAX=1 PATTERNKV_ACTIVE_BATCH_CACHE=1 PATTERNKV_SYSTEM_PROFILE=0 \
/data/zypan/.local/share/mamba/envs/patternkv/bin/python <inline torch profiler one-step decode>
```
