# PatternKV LongBench Paper v2 Report

Status: SMOKE PASS
Expected total records: 9
Actual total records: 9
Absolute delta (kivi_paper_g128 - fp16): -1.47
Quality retention percent: 96.5922

## Method Averages

| method | avg_normalized |
| --- | ---: |
| fp16 | 43.1367 |
| kivi_paper_g128 | 41.6667 |
| patternkv_paper | 42.2233 |

## Per Task

| method | task | samples | failures | empty | score | metric | avg input | avg output |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| fp16 | qasper | 1 | 0 | 0 | 29.41 | qa_f1 | 4034.0 | 34.0 |
| fp16 | passage_retrieval_en | 1 | 0 | 0 | 0.0 | retrieval | 8183.0 | 32.0 |
| fp16 | lcc | 1 | 0 | 0 | 100.0 | code_sim | 8168.0 | 64.0 |
| kivi_paper_g128 | qasper | 1 | 0 | 0 | 25.0 | qa_f1 | 4034.0 | 42.0 |
| kivi_paper_g128 | passage_retrieval_en | 1 | 0 | 0 | 0.0 | retrieval | 8183.0 | 32.0 |
| kivi_paper_g128 | lcc | 1 | 0 | 0 | 100.0 | code_sim | 8168.0 | 64.0 |
| patternkv_paper | qasper | 1 | 0 | 0 | 26.67 | qa_f1 | 4034.0 | 30.0 |
| patternkv_paper | passage_retrieval_en | 1 | 0 | 0 | 0.0 | retrieval | 8183.0 | 32.0 |
| patternkv_paper | lcc | 1 | 0 | 0 | 100.0 | code_sim | 8168.0 | 64.0 |

## PatternKV Bit Accounting

```json
{
  "avg_input_tokens": 6795.0,
  "avg_output_tokens": 42.0,
  "avg_v_mask_mean": 0.78889,
  "cuda_compact_transfer_bits": {
    "assumptions": "Same payload and scale/min, but CUDA wrapper compact transfer dtypes: K assignment int16, V assignment uint8.",
    "centroid_fp16_bits_per_dim_amortized_per_k_or_v": 0.0763,
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
    "centroid_fp16_bits_per_dim_amortized_per_k_or_v": 0.0763,
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
