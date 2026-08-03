# AIME24 ETA Report

Status: NOT RUN.

Reason: `MODEL_PATH` is not set and no local DeepSeek-R1-Distill-Llama-8B candidate was found.

Prepared ETA command:

```bash
MODEL_PATH=/path/to/DeepSeek-R1-Distill-Llama-8B \
MAX_NEW_TOKENS=32768 \
bash scripts/run_aime24_patternkv_eta_8gpu.sh
```

Without ETA measurements, the framework cannot honestly estimate whether the 180-task run fits inside 10 hours. Recommended first execution after model path is available: ETA, then choose `NUM_SAMPLES=2` only if the estimated total is within the requested budget.
