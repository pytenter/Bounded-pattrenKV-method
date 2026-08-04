# PatternKV LongBench 8x50 Report

Status: FULL RUN PASS
Expected total records: 400
Actual total records: 400
Absolute delta (kivi_official - kivi_official): 0.0
Quality retention percent: 100.0

## Method Averages

| method | avg_normalized |
| --- | ---: |
| kivi_official | 49.4738 |

## Per Task

| method | task | samples | failures | empty | score | metric | avg input | avg output |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| kivi_official | qasper | 50 | 0 | 0 | 40.37 | qa_f1 | 5146.64 | 35.8 |
| kivi_official | multifieldqa_en | 50 | 0 | 0 | 51.28 | qa_f1 | 6344.64 | 26.62 |
| kivi_official | hotpotqa | 50 | 0 | 0 | 22.42 | qa_f1 | 7864.46 | 27.1 |
| kivi_official | 2wikimqa | 50 | 0 | 0 | 38.71 | qa_f1 | 6257.58 | 12.62 |
| kivi_official | gov_report | 50 | 0 | 0 | 34.65 | rouge_l | 7308.98 | 463.92 |
| kivi_official | trec | 50 | 0 | 0 | 70.0 | classification | 5885.34 | 64.0 |
| kivi_official | passage_retrieval_en | 50 | 0 | 0 | 76.0 | retrieval | 8182.94 | 11.48 |
| kivi_official | lcc | 50 | 0 | 0 | 62.36 | code_sim | 2983.78 | 64.0 |

## PatternKV Bit Accounting

```json
{
  "avg_input_tokens": 0.0,
  "avg_output_tokens": 0.0,
  "avg_v_mask_mean": null,
  "cuda_compact_transfer_bits": {
    "assumptions": "Same payload and scale/min, but CUDA wrapper compact transfer dtypes: K assignment int16, V assignment uint8.",
    "centroid_fp16_bits_per_dim_amortized_per_k_or_v": 512.0,
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
    "centroid_fp16_bits_per_dim_amortized_per_k_or_v": 512.0,
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
