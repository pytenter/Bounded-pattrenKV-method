# Compatibility Changes

## Summary

All changes are compatibility/instrumentation/support changes. No PatternKV algorithmic math was replaced.

## Files

- `.gitignore`: ignore local wheels, build outputs, Python caches, and test caches. Does not affect runtime.
- `patternkv.py`: minimal package import anchor so `import patternkv` works after editable install. Does not affect model execution.
- `models/llama_patternkv.py`: changed `cuml` and `cupy` top-level imports to optional imports. Real PatternKV path uses the PyTorch batched kmeans helpers in this file; RAPIDS/CuPy are not called in the audited path.
- `quant/setup.py`: add runtime rpath to the active torch library directory at build time. This lets `import patternkv_gemv` work without first importing torch or exporting LD_LIBRARY_PATH. Does not affect CUDA kernels.
- `tests/test_quant_extension.py`: added random tensor unit validation for pack/dequant and custom CUDA GEMV.
- `scripts/run_smoke.py`: added unified FP16/PatternKV smoke runner and read-only cache/stat collection.

## Environment deviations

- Actual GPU is RTX 3090 / SM86, not requested RTX 4090 / SM89.
- Used `TORCH_CUDA_ARCH_LIST=8.6` for extension compilation because the visible GPU is RTX 3090.
- `flash-attn==2.6.3` was required by the official PatternKV flash attention path. The package installer hit a cross-device link issue, so the release wheel was downloaded manually and installed locally.
- `attributedict==0.3.0` was installed with `--no-deps` to avoid pulling unrelated tox/coverage developer dependencies.
