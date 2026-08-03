# PatternKV LongBench 8x50 Report

Status: FULL RUN PASS
Expected total records: 1200
Actual total records: 1200
Absolute delta (patternkv - fp16): -1.4137
Quality retention percent: 97.1911

## Method Averages

| method | avg_normalized |
| --- | ---: |
| fp16 | 50.33 |
| patternkv | 48.9163 |
| kivi | 30.075 |

## Per Task

| method | task | samples | failures | empty | score | metric | avg input | avg output |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| fp16 | qasper | 50 | 0 | 0 | 41.08 | qa_f1 | 5146.64 | 35.4 |
| fp16 | multifieldqa_en | 50 | 0 | 0 | 51.37 | qa_f1 | 6344.64 | 27.82 |
| fp16 | hotpotqa | 50 | 0 | 0 | 22.61 | qa_f1 | 7864.46 | 28.58 |
| fp16 | 2wikimqa | 50 | 0 | 0 | 40.13 | qa_f1 | 6257.58 | 13.08 |
| fp16 | gov_report | 50 | 0 | 0 | 35.47 | rouge_l | 7308.98 | 465.94 |
| fp16 | trec | 50 | 0 | 0 | 72.0 | classification | 5885.34 | 64.0 |
| fp16 | passage_retrieval_en | 50 | 0 | 0 | 76.0 | retrieval | 8182.94 | 9.32 |
| fp16 | lcc | 50 | 0 | 0 | 63.98 | code_sim | 2983.78 | 64.0 |
| patternkv | qasper | 50 | 0 | 0 | 41.59 | qa_f1 | 5146.64 | 35.66 |
| patternkv | multifieldqa_en | 50 | 0 | 0 | 49.21 | qa_f1 | 6344.64 | 32.3 |
| patternkv | hotpotqa | 50 | 0 | 0 | 22.26 | qa_f1 | 7864.46 | 26.52 |
| patternkv | 2wikimqa | 50 | 0 | 0 | 37.82 | qa_f1 | 6257.58 | 13.04 |
| patternkv | gov_report | 50 | 0 | 0 | 34.04 | rouge_l | 7308.98 | 502.18 |
| patternkv | trec | 50 | 0 | 0 | 72.0 | classification | 5885.34 | 64.0 |
| patternkv | passage_retrieval_en | 50 | 0 | 0 | 71.67 | retrieval | 8182.94 | 21.76 |
| patternkv | lcc | 50 | 0 | 0 | 62.74 | code_sim | 2983.78 | 64.0 |
| kivi | qasper | 50 | 0 | 0 | 23.77 | qa_f1 | 5146.64 | 34.52 |
| kivi | multifieldqa_en | 50 | 0 | 0 | 30.74 | qa_f1 | 6344.64 | 35.62 |
| kivi | hotpotqa | 50 | 0 | 0 | 10.15 | qa_f1 | 7864.46 | 30.58 |
| kivi | 2wikimqa | 50 | 0 | 0 | 21.0 | qa_f1 | 6257.58 | 21.14 |
| kivi | gov_report | 50 | 0 | 0 | 15.7 | rouge_l | 7308.98 | 497.04 |
| kivi | trec | 50 | 0 | 0 | 64.0 | classification | 5885.34 | 64.0 |
| kivi | passage_retrieval_en | 50 | 0 | 0 | 41.48 | retrieval | 8182.94 | 32.0 |
| kivi | lcc | 50 | 0 | 0 | 33.76 | code_sim | 2983.78 | 64.0 |

## PatternKV Bit Accounting

```json
{
  "avg_input_tokens": 6246.8,
  "avg_output_tokens": 94.93,
  "avg_v_mask_mean": 0.794241,
  "cuda_compact_transfer_bits": {
    "assumptions": "Same payload and scale/min, but CUDA wrapper compact transfer dtypes: K assignment int16, V assignment uint8.",
    "centroid_fp16_bits_per_dim_amortized_per_k_or_v": 0.0824,
    "k_assignment_actual_cache_bits_per_dim": 0.125,
    "k_payload_bits_per_dim": 2.0,
    "k_scale_min_bits_per_dim": 0.25,
    "v_assignment_actual_cache_bits_per_dim": 0.0625,
    "v_mask_actual_cache_bits_per_dim": 0.0625,
    "v_payload_bits_per_dim": 2.0,
    "v_scale_min_bits_per_dim": 0.25
  },
  "current_cache_layout_bits": {
    "assumptions": "2-bit packed payload, group_size=128, head_dim=128, FP16 scale/min, current Python cache stores K/V assignments as torch.long and V mask as uint8.",
    "centroid_fp16_bits_per_dim_amortized_per_k_or_v": 0.0824,
    "k_assignment_actual_cache_bits_per_dim": 0.5,
    "k_payload_bits_per_dim": 2.0,
    "k_scale_min_bits_per_dim": 0.25,
    "v_assignment_actual_cache_bits_per_dim": 0.5,
    "v_mask_actual_cache_bits_per_dim": 0.0625,
    "v_payload_bits_per_dim": 2.0,
    "v_scale_min_bits_per_dim": 0.25
  }
}
```

## Issues

No integrity issues found.
