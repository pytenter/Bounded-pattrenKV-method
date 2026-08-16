# Test Environment

- Command wrapper: `/data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv`
- Python executable: `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`
- PyTorch: `2.4.1+cu124`
- CUDA runtime: `12.4`
- CUDA available: `True`
- pytest: `9.1.1`

The earlier `/usr/bin/python3` shell was not the project environment and lacked `torch`/`pytest`.
