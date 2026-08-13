# Validation Summary

Generated: 2026-08-13 11:04:09 +08

## Build State

No CUDA extension rebuild was required in S5A-4 because this phase did not change production code or kernel math. The loaded extension is the S5A-3 build:

- path: `/data/zypan/Bounded-pattrenKV-pseudodecode-3090/quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so`
- sha256: `c725036c10ef700c5c668d320934adb366e0069d44cefc0dab6a5c77fc7aa31e`
- mtime: `1786588657`

Environment:

- Python: `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`
- PyTorch: `2.4.1+cu124`
- CUDA runtime: `12.4`
- GPU used: `NVIDIA GeForce RTX 3090`
- physical GPU id: `5`
- driver: `580.173.02`

## Tooling

- `ncu` in PATH: NO
- `nsys` in PATH: NO
- `/usr/local/cuda-12.4/bin/nvcc`: available
- `/usr/local/cuda-12.4/bin/cuobjdump`: available
- `/usr/local/cuda-12.4/bin/nvdisasm`: available
- `/usr/local/cuda-12.4/bin/ptxas`: available

Fallback evidence generated:

```text
cuobjdump --dump-resource-usage
cuobjdump --dump-sass
SASS instruction category counts
static source addressing audit
S5A-3 CUDA Event latency reference
```

The raw 13 MB SASS dump was not retained in git; summarized SASS counts are stored in `sass_instruction_counts.csv` and `sass_instruction_counts.json`.

## Checks

Compile and patch hygiene:

```text
python -m compileall bench models quant scripts tests
PASS

git diff --check
PASS
```

Full regression suite:

```text
646 passed in 14.26s
```

## Scope Guard

- Algorithm changed: NO
- Kernel math changed: NO
- Production backend changed: NO
- Value capacity path changed: NO
- K capacity re-enabled: NO
- Full AIME24/AIME25/GPQA evaluation: NO
- Real model E2E decode: NO
- vLLM/SGLang/CUDA VMM/concurrency: NO

## Classification

`K_STRIDE_REGRESSION_PARTIALLY_SUPPORTED`

Dominant measured/static mechanism: `ADDRESS_ARITHMETIC_DOMINATED`, confidence `MEDIUM`.

`PHYSICAL_CAPACITY_SCAN_HYPOTHESIS=REJECTED`.

NEXT_TASK: `V_ONLY_CAPACITY_FINAL_SYSTEM_BENCHMARK`
