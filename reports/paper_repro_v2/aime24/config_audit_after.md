# AIME24 Config Audit After

Generated: 2026-08-04

Status: PARTIAL PASS.

Implemented:

- `datasets/aime/aime24.jsonl`: 30 normalized AIME24 problems, `problem_id=0..29`.
- `datasets/aime/aime24_metadata.json`: source, split, revision, checksum.
- `bench/aime_answer_parser.py`: robust AIME 0..999 parser.
- `bench/aime_utils.py`: manifest, paired seeds, config hash, resume checks, majority/paired helpers.
- `bench/bench_aime24_patternkv.py`: AIME runner for `fp16`, `kivi_paper_g128`, `patternkv_paper`.
- `scripts/run_aime24_patternkv_smoke.sh`: 1 problem × 1 sample × 3 methods, smoke-only output.
- `scripts/run_aime24_patternkv_eta_8gpu.sh`: 3 problems × 1 sample × 3 methods ETA.
- `scripts/run_aime24_patternkv_budget_n2_8gpu.sh`: 8GPU staged full run, no tensor parallel.
- `scripts/summarize_aime24_results.py`: Avg@N, strict_avg, majority, paired comparison, length buckets.
- AIME tests covering parser, manifest, seed pairing, resume, summary, and paper config.

Dataset:

- project SHA256: `07ec3f0c489406676be9d6057e2f97c9c32bc18e856d13df1d05c76724cbb08f`
- source original SHA256: `71ddb0950cc39f58767f7217f328cd759cfefcf70ca91cfcd5c777155a5f9b63`

Blocking issue:

- `MODEL_PATH` is not set and no local DeepSeek-R1-Distill-Llama-8B model was found. Therefore AIME smoke and ETA were not launched, and no full 180-task run was started.
