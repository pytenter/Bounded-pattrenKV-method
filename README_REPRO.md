# PatternKV RTX 4090 Smoke Reproduction Notes

This workspace reproduced the official PatternKV code path in `/data/zypan/PatternKV-repro` on the currently visible hardware.

Important: the server exposes RTX 3090 GPUs, not RTX 4090. CUDA extension and smoke results are therefore SM86 results, not SM89/4090 validation.

## Environment

```bash
/data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python --version
CUDA_VISIBLE_DEVICES=0 /data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## Key Commands

```bash
CUDA_VISIBLE_DEVICES=0 /data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python tests/test_quant_extension.py

CUDA_VISIBLE_DEVICES=0 /data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python scripts/run_smoke.py \
  --model-path /data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct \
  --method fp16 --device cuda:0 --dtype float16 --max-new-tokens 160 \
  --min-input-tokens 256 --output-json results/smoke_fp16.json

CUDA_VISIBLE_DEVICES=0 PATTERNKV_DEBUG_STATS=1 /data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python scripts/run_smoke.py \
  --model-path /data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct \
  --method patternkv --device cuda:0 --dtype float16 \
  --k-bits 2 --v-bits 2 --group-size 128 --residual-length 128 \
  --num-k-base 32 --num-v-base 32 --max-new-tokens 160 \
  --output-json results/smoke_patternkv.json
```

Reports are in `reports/`, outputs in `results/`, logs in `logs/`.
