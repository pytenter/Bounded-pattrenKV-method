# AIME25 Full Causal25 Quality Summary

Generated from compact JSON results under `results/aime25_full_causal25_quality_v100/formal/`.

- Handoff log: `run/aime25_full_causal25_quality_v100/logs/patternkv_gpu0_3_handoff.tmux.log`
- Scheduler completion: `{"completed": 90, "event": "scheduler_complete", "timestamp": "2026-08-25T15:46:37+0800"}`

## Aggregate

| Dataset | Method | Backend | Model | Progress | Correct | Accuracy | Failed | EOS stop | Length stop | Last result |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| AIME25 | FP16 | fp16 | DeepSeek-R1-Distill-Llama-8B | 90/90 | 30/90 | 33.33% | 0 | 85 | 5 | 2026-08-21 17:50:33 |
| AIME25 | PATTERN_BASE | patternkv_paper | DeepSeek-R1-Distill-Llama-8B | 90/90 | 21/90 | 23.33% | 0 | 81 | 9 | 2026-08-25 15:46:31 |
| AIME25 | CAUSAL_V4_25 | patternkv | DeepSeek-R1-Distill-Llama-8B | 90/90 | 27/90 | 30.00% | 0 | 82 | 8 | 2026-08-24 22:30:55 |

## Seed Breakdown

| Method | Seed | Progress | Correct | Accuracy | Failed | EOS stop | Length stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 42 | 30/30 | 10/30 | 33.33% | 0 | 26 | 4 |
| FP16 | 43 | 30/30 | 11/30 | 36.67% | 0 | 29 | 1 |
| FP16 | 44 | 30/30 | 9/30 | 30.00% | 0 | 30 | 0 |
| PATTERN_BASE | 42 | 30/30 | 10/30 | 33.33% | 0 | 29 | 1 |
| PATTERN_BASE | 43 | 30/30 | 6/30 | 20.00% | 0 | 25 | 5 |
| PATTERN_BASE | 44 | 30/30 | 5/30 | 16.67% | 0 | 27 | 3 |
| CAUSAL_V4_25 | 42 | 30/30 | 10/30 | 33.33% | 0 | 25 | 5 |
| CAUSAL_V4_25 | 43 | 30/30 | 10/30 | 33.33% | 0 | 29 | 1 |
| CAUSAL_V4_25 | 44 | 30/30 | 7/30 | 23.33% | 0 | 28 | 2 |

## Files

- `summary.csv`: aggregate method-level table.
- `seed_breakdown.csv`: seed-level table.
