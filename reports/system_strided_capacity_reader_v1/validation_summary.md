# Validation Summary

- Build command: `CUDA_VISIBLE_DEVICES=1 /data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pip install --no-build-isolation -e .`
- Loaded extension: `/data/zypan/Bounded-pattrenKV-pseudodecode-3090/quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so`
- Loaded extension mtime: `1786584370`
- Loaded extension SHA256: `e6aa20c011a4a100d07abd3656dc21677bdc3122d38c158188ff56608b7eec36`
- Python: `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`
- PyTorch: `2.4.1+cu124`
- CUDA used by PyTorch: `12.4`
- Driver: `580.173.02`
- Physical benchmark GPU: `1`
- GPU model: `NVIDIA GeForce RTX 3090`

Validation commands:

- `PYTHONPATH=/data/zypan/Bounded-pattrenKV-pseudodecode-3090/quant:/data/zypan/Bounded-pattrenKV-pseudodecode-3090 /data/zypan/.local/share/mamba/envs/patternkv/bin/python -m compileall bench models quant scripts tests`: PASS
- `git diff --check`: PASS
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data/zypan/Bounded-pattrenKV-pseudodecode-3090/quant:/data/zypan/Bounded-pattrenKV-pseudodecode-3090 /data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest -q tests/test_strided_capacity_value_reader.py`: `23 passed`
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data/zypan/Bounded-pattrenKV-pseudodecode-3090/quant:/data/zypan/Bounded-pattrenKV-pseudodecode-3090 /data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest -q`: `592 passed, 1 warning`

Not run:

- Full AIME24: NO
- AIME25: NO
- E2E model decode: NO
- vLLM: NO
- SGLang: NO
- CUDA VMM: NO
