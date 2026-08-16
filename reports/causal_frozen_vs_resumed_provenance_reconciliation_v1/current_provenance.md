# Current Provenance

Current report path: `reports/paper_baseline_system_comparison_v1/`.

Stored current environment reports HEAD `50a9a2748789dc4313c880f5f7643ae6f1b8d256`, Python `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`, torch `2.4.1+cu124`, CUDA runtime `12.4`, and physical GPU 1 UUID `GPU-624f86d9-284b-cb46-a671-51d77559dab6`.

The current paper runner `bench/paper_baseline_system_comparison.py` launches one worker subprocess per row and sets `CUDA_VISIBLE_DEVICES`, `PAPER_BASELINE_GPU_UUID`, `PATTERNKV_SELECTIVE_PREFILL_LOGITS=1`, `PATTERNKV_ACTIVE_BATCH_CACHE=1`, and `PATTERNKV_SYSTEM_PROFILE=0` inside the worker. It does not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, unlike the frozen freeze runner.

Direct current worker reproduction under the frozen env reproduces the fast regime and B8 capacity. Direct current B8 without `PYTORCH_CUDA_ALLOC_CONF` reproduces the current OOM signature.
