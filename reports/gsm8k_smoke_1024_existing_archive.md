# GSM8K Smoke Existing Result Archive

## Important Finding

The currently complete three-method GSM8K smoke under `results/gsm8k/smoke` records `max_new_tokens=1024`, not 512. It was produced during opportunistic recovery using GPUs 1/3/5/7 and is archived as existing evidence, not as the requested official GPU4-7 rerun.

## Summary

- fp16: rows=50, accuracy=82.0%, length_truncated=1, errors=0, max_new_tokens=1024
- kivi: rows=50, accuracy=36.0%, length_truncated=20, errors=0, max_new_tokens=1024
- patternkv: rows=50, accuracy=84.0%, length_truncated=1, errors=0, max_new_tokens=1024

## Archive Paths

- `results/gsm8k/smoke_1024_existing_mixed_gpu`: files=15, bytes=1205446
- `logs/gsm8k/smoke_1024_existing_mixed_gpu`: files=10, bytes=20322
- `run/gsm8k/smoke_1024_existing_mixed_gpu`: files=34, bytes=8390
- `reports/gsm8k_smoke_1024_existing_mixed_gpu.json`: bytes=3040, sha256=914c5d10114ca5dc73293857b6408b58bc1db6f96ec3921f6e3f66eaf70bd009
- `reports/gsm8k_smoke_1024_existing_mixed_gpu.md`: bytes=789, sha256=b3cf75af941cbecf17099cb8fe873364bcfc19a8795b34741f323a735f8ffceb
