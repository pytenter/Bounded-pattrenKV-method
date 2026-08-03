# PatternKV LongBench 8x50 Report

Status: SMOKE PASS
Expected total records: 32
Actual total records: 32
FP16 avg_normalized: 53.3225
PatternKV avg_normalized: 50.1137
Absolute delta: -3.2088
Quality retention percent: 93.9823

## Per Task

| method | task | samples | failures | empty | score | metric | avg input | avg output |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| fp16 | qasper | 2 | 0 | 0 | 40.63 | qa_f1 | 3699.5 | 28.5 |
| fp16 | multifieldqa_en | 2 | 0 | 0 | 48.86 | qa_f1 | 4873.5 | 34.5 |
| fp16 | hotpotqa | 2 | 0 | 0 | 28.64 | qa_f1 | 8183.0 | 32.0 |
| fp16 | 2wikimqa | 2 | 0 | 0 | 20.0 | qa_f1 | 7039.5 | 6.0 |
| fp16 | gov_report | 2 | 0 | 0 | 38.45 | rouge_l | 8183.0 | 494.0 |
| fp16 | trec | 2 | 0 | 0 | 100.0 | classification | 6537.0 | 64.0 |
| fp16 | passage_retrieval_en | 2 | 0 | 0 | 50.0 | retrieval | 8183.0 | 18.5 |
| fp16 | lcc | 2 | 0 | 0 | 100.0 | code_sim | 5481.5 | 64.0 |
| patternkv | qasper | 2 | 0 | 0 | 34.55 | qa_f1 | 3699.5 | 29.5 |
| patternkv | multifieldqa_en | 2 | 0 | 0 | 46.39 | qa_f1 | 4873.5 | 34.5 |
| patternkv | hotpotqa | 2 | 0 | 0 | 17.98 | qa_f1 | 8183.0 | 32.0 |
| patternkv | 2wikimqa | 2 | 0 | 0 | 20.0 | qa_f1 | 7039.5 | 6.0 |
| patternkv | gov_report | 2 | 0 | 0 | 31.99 | rouge_l | 8183.0 | 512.0 |
| patternkv | trec | 2 | 0 | 0 | 100.0 | classification | 6537.0 | 64.0 |
| patternkv | passage_retrieval_en | 2 | 0 | 0 | 50.0 | retrieval | 8183.0 | 18.5 |
| patternkv | lcc | 2 | 0 | 0 | 100.0 | code_sim | 5481.5 | 64.0 |

## PatternKV Bit Accounting

```json
{
  "avg_input_tokens": 6522.5,
  "avg_output_tokens": 95.06,
  "avg_v_mask_mean": 0.793573,
  "cuda_compact_transfer_bits": {
    "assumptions": "Same payload and scale/min, but CUDA wrapper compact transfer dtypes: K assignment int16, V assignment uint8.",
    "centroid_fp16_bits_per_dim_amortized_per_k_or_v": 0.0789,
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
    "centroid_fp16_bits_per_dim_amortized_per_k_or_v": 0.0789,
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
