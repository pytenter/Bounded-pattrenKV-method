# AIME24 Config Audit After

Generated: 2026-08-04

Status: COMPLETE for the resource-bounded AIME24 `num_samples=2` run.

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

Full run:

- Completed on August 5-6, 2026 (Beijing time) with 8 independent V100 workers.
- Planned: `30 problems × 2 samples × 3 methods = 180`.
- Completed: `180/180`.
- Runtime errors: `0`.
- CUDA OOM failures: `0`.
- Aggregate report: `reports/paper_repro_v2/aime24/results_summary.md`.
- Detailed report: `reports/paper_repro_v2/aime24/full_run_report.md`.

The run is resource-bounded at `num_samples=2`; it is not the paper's
`Avg@8`/`Maj@8` evaluation.
