# Validation Summary

Generated: 2026-08-13 10:21:02 +08

## Build

Command:

```bash
CUDA_VISIBLE_DEVICES=5 /data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pip install --no-build-isolation -e .
```

Loaded extension:

- path: `/data/zypan/Bounded-pattrenKV-pseudodecode-3090/quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so`
- sha256: `1db0cfcd6de556426efee92dd1ee27659487c2083ca9e8ccba9d2633af1a9637`
- mtime: `1786585795`

Environment:

- Python: `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`
- PyTorch: `2.4.1+cu124`
- CUDA: `12.4`
- GPU: `NVIDIA GeForce RTX 3090`

## Checks

Targeted integration tests:

```text
tests/test_capacity_cache_integration.py tests/test_strided_capacity_value_reader.py
52 passed in 6.40s
```

Focused regression rerun after `int32` compact metadata compatibility fix:

```text
6 passed in 2.72s
```

Full regression suite:

```text
621 passed in 13.15s
```

Compile and patch hygiene:

```text
python -m compileall bench models quant scripts tests
PASS

git diff --check
PASS
```

## Scope Guard

- Full AIME24/AIME25/GPQA evaluation: not run in this phase.
- vLLM integration: not used.
- SGLang integration: not used.
- CUDA VMM: not used.
- Page-native reader: not used.
- Experimental GQA V backend: not used.
- Selector, quantization formula, centroids, sink/recent/residual/group settings, and attention math: unchanged.
