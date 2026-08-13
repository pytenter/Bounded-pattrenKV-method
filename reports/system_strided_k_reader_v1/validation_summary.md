# Validation Summary

Generated: 2026-08-13 10:40:10 +08

## Build

Command:

```bash
CUDA_VISIBLE_DEVICES=5 /data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pip install --no-build-isolation -e .
```

Loaded extension:

- path: `/data/zypan/Bounded-pattrenKV-pseudodecode-3090/quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so`
- sha256: `c725036c10ef700c5c668d320934adb366e0069d44cefc0dab6a5c77fc7aa31e`
- mtime: `1786588657`

Environment:

- Python: `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`
- PyTorch: `2.4.1+cu124`
- CUDA: `12.4`
- GPU: `NVIDIA GeForce RTX 3090`
- physical GPU id: `5`
- NCU available: `NO`

## Checks

Targeted K reader tests:

```text
tests/test_strided_k_reader.py
25 passed in 4.41s
```

Full regression suite:

```text
646 passed in 13.00s
```

Compile and patch hygiene:

```text
python -m compileall bench models quant scripts tests
PASS

git diff --check
PASS
```

## Benchmark

Command:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/data/zypan/Bounded-pattrenKV-pseudodecode-3090/quant:/data/zypan/Bounded-pattrenKV-pseudodecode-3090 /data/zypan/.local/share/mamba/envs/patternkv/bin/python bench/bench_strided_k_reader.py --contexts 8192 16384 24576 32768 --warmup 30 --iters 200 --rounds 7 --physical-gpu 5
```

Result:

- correctness: PASS, 13/13 cases
- max_abs: `0.0`
- relative_L2 max: `0.0`
- cosine min: `0.9999998807907104`
- NaN/Inf: `0/0`
- historical materialization: `0` calls, `0` bytes
- logical-token-only counter: PASS
- 32K overhead: `33.68%`
- classification: `STRIDED_K_READER_NOT_SUPPORTED`

## Scope Guard

- Full K capacity cache integration: not done.
- Value capacity path: not changed.
- Full AIME24/AIME25/GPQA evaluation: not run.
- Real model E2E decode: not run.
- vLLM integration: not used.
- SGLang integration: not used.
- CUDA VMM: not used.
- Page-native K reader/page table: not used.
- Experimental GQA redesign: not used.
- Selector, quantization formula, centroids, sink/recent/residual/group settings, and attention math: unchanged.
