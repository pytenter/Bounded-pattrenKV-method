# Exact Commands

```bash
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline --decorate
git diff --stat
git diff --check
nvidia-smi
```

```bash
CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATTERNKV_FIXED_SPLIT_SOFTMAX=1 PATTERNKV_ACTIVE_BATCH_CACHE=1 PATTERNKV_SYSTEM_PROFILE=0 /data/zypan/.local/share/mamba/envs/patternkv/bin/python scripts/full_model_scaling_decode_only_protocol_repair.py --phases all
```
