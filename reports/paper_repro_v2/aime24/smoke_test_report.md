# AIME24 Smoke Test Report

Status: NOT RUN.

Reason: `MODEL_PATH` is not set and no local DeepSeek-R1-Distill-Llama-8B candidate was found. The framework refuses to substitute Llama-3.1-8B-Instruct.

Prepared smoke command:

```bash
MODEL_PATH=/path/to/DeepSeek-R1-Distill-Llama-8B \
MAX_NEW_TOKENS=1024 \
bash scripts/run_aime24_patternkv_smoke.sh
```

Smoke output directory will be `results/paper_repro_v2/aime24_smoke`, separate from formal results.
