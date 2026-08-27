# Reproduce

Prepare manifest:

```bash
python scripts/run_gsm8k_selector_components_pilot.py prepare --selected-gpus 1 4 --forbidden-gpus 2 3
```

Run one shard:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_gsm8k_selector_components_pilot.py run --phase pilot --method importance_only_v4_25 --physical-gpu-id 1 --worker-index 0 --num-workers 1
```
