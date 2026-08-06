# AIME24 Full Run Report

## Executive Summary

This resource-bounded AIME24 run evaluates three paper-v2 methods on the same
30 AIME24 problems with two paired samples per problem:

- `fp16`
- `kivi_paper_g128`
- `patternkv_paper`

The run completed all `180/180` planned tasks on 8 independent V100 workers.
There were no CUDA out-of-memory failures or runtime errors. The results are
`Avg@2`-style results and must not be reported as the paper's `Avg@8` or
`Maj@8` results.

## Main Results

| method | planned | completed | valid | correct | Avg@2 | strict_avg | parse rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp16` | 60 | 60 | 59 | 29 | 49.1525% | 48.3333% | 98.3333% |
| `kivi_paper_g128` | 60 | 60 | 56 | 12 | 21.4286% | 20.0000% | 93.3333% |
| `patternkv_paper` | 60 | 60 | 54 | 13 | 24.0741% | 21.6667% | 90.0000% |

`Avg@2` is computed over valid parsed responses. `strict_avg` uses all 60
planned samples as the denominator, so it includes parser failures.

## Paired Comparisons

All comparisons use the same problem/sample keys and therefore preserve the
paired seeds.

| comparison | paired n | accuracy difference | both correct | only left correct | only right correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| `patternkv_paper - kivi_paper_g128` | 60 | +1.6667 pp | 8 | 5 | 4 |
| `patternkv_paper - fp16` | 60 | -26.6667 pp | 11 | 2 | 18 |
| `kivi_paper_g128 - fp16` | 60 | -28.3333 pp | 9 | 3 | 20 |

On this run, PatternKV is slightly ahead of KIVI in paired accuracy, but both
INT2 methods are substantially below FP16 on AIME24.

## Reliability and Stopping Behavior

| method | EOS stop | length stop | OOM | other errors | parser failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` | 57 | 3 | 0 | 0 | 1 |
| `kivi_paper_g128` | 50 | 10 | 0 | 0 | 4 |
| `patternkv_paper` | 52 | 8 | 0 | 0 | 6 |

The generation budget was `max_new_tokens=32768`. The higher length-stop rate
for the two INT2 methods is an important part of the observed quality gap and
should not be omitted from interpretation.

## Generation and Runtime Statistics

| method | average generated tokens | average task wall time | p95 task wall time | average tokens/s |
| --- | ---: | ---: | ---: | ---: |
| `fp16` | 13,220.93 | 925.4553 s | 2,473.3396 s | 18.4594 |
| `kivi_paper_g128` | 16,430.63 | 1,028.2430 s | 2,035.3173 s | 15.9406 |
| `patternkv_paper` | 14,769.17 | 820.5979 s | 2,179.8863 s | 19.5939 |

Wall-clock schedule, in Beijing time:

- `fp16`: 17:13:17 to 20:10:13 on August 5, 2026
- `kivi_paper_g128`: 20:10:13 to 22:55:45 on August 5, 2026
- `patternkv_paper`: 22:55:45 on August 5 to 01:32:43 on August 6, 2026
- summary generation completed at approximately 01:33 on August 6, 2026

The end-to-end GPU run therefore took approximately 8 hours 20 minutes,
excluding the earlier launcher setup interval.

## Protocol

- Dataset: normalized AIME24, 30 problems, `problem_id=0..29`
- Samples: 2 per problem, paired seeds
- Workers: 8 independent single-GPU workers; no tensor parallelism
- Model: local DeepSeek-R1-Distill-Llama-8B checkpoint
- Prompt: DeepSeek-R1 recommended math prompt with boxed final answer
- Sampling: `temperature=0.6`, `top_p=0.95`, sampling enabled
- Maximum generation: `32768` new tokens
- KV configuration:
  - KIVI: 2-bit K/V, group size 128, residual length 128
  - PatternKV: 2-bit K/V, group size 128, 32 K bases, 32 V bases

Full configuration details are in `experiment_protocol.md`, and the machine
readable aggregate is `results_summary.json`.

## Interpretation Limits

This is a resource-bounded `n=2` run. It is sufficient to check the current
implementation and compare the observed methods under a fixed protocol, but it
is not a replacement for the paper's larger-sample evaluation. The parser
failure and length-stop counts also mean that valid-response accuracy and
planned-task accuracy should be reported together.

## Artifacts

- Runner: `bench/bench_aime24_patternkv.py`
- Launcher: `scripts/run_aime24_patternkv_budget_n2_8gpu.sh`
- Summarizer: `scripts/summarize_aime24_results.py`
- Aggregate Markdown: `results_summary.md`
- Aggregate JSON: `results_summary.json`
- Per-sample JSON: `results/paper_repro_v2/aime24_budget_n2/`
