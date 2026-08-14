# CAUSAL_V4_25 Generalization

## Frozen Method
CAUSAL_V4_25 is registered as PatternKV segmented rolling K2/V mixed precision with selector `causal_v4`, V4 budget 25%, group 128, sink 16, recent 128, residual 128, num_k_base 32, num_v_base 32. No selector formula or benchmark-specific tuning was changed.

## Protocol Audit
BASELINE_REUSE_STATUS = `SAFE_TO_REUSE`. Historical GSM8K and LongBench baseline outputs are complete and sample-aligned for `fp16`, `kivi_paper_g128`, and `patternkv_paper`. See `protocol_audit.json`.

## Effective Bitwidth
```json
{
  "fp16": {
    "K_bits": 16,
    "V_bits": "16",
    "payload_bits_per_KV_scalar": 16.0,
    "effective_quantized_region_bits": "16.0 payload; no quant metadata",
    "residual_sink_recent": "not applicable"
  },
  "kivi_paper_g128": {
    "K_bits": 2,
    "V_bits": "2",
    "group_size": 128,
    "scale_zero_metadata": "FP16 scale + FP16 min/zero per group",
    "theoretical_quantized_region_bits": "2 + 32/128 = 2.25 bit/scalar",
    "residual_length": 128
  },
  "patternkv_paper": {
    "K_bits": 2,
    "V_bits": "2",
    "group_size": 128,
    "num_k_base": 32,
    "num_v_base": 32,
    "pattern_group": 128,
    "scale_zero_metadata": "FP16 scale + FP16 min/zero per group plus pattern assignments/centroids in implementation",
    "theoretical_quantized_region_bits": "2 + 32/128 = 2.25 bit/scalar before pattern assignment/centroid physical overhead",
    "residual_length": 128
  },
  "causal_v4_25": {
    "K_bits": 2,
    "V_bit_distribution": "75% V2 / 25% V4 in full V windows",
    "V4_ratio": 0.25,
    "group_size": 128,
    "sink": 16,
    "recent": 128,
    "residual_length": 128,
    "selector": "causal_v4",
    "metadata_dtype": "V precision gate observed as torch.uint8 in smoke records; assignments torch.int64 in Python tensors",
    "raw_payload_bits": "K payload 2; V average payload 2*0.75 + 4*0.25 = 2.5; combined K/V payload average = 2.25 bit/scalar",
    "level_a_effective_budget_from_release": "2.50048828125 bit/element including INT2/INT4 payload, FP16 scale/min, mixed V2/V4 precision metadata; excludes sink/recent FP16, centroids, assignment tensors, allocator overhead",
    "physical_storage_note": "smoke records include packed_payload_bytes, scale_min_bytes, fp16_residual_bytes, assignment_bytes, mask_bytes, centroid_bytes; allocator-level full physical bitwidth not finalized"
  }
}
```

## GSM8K
CAUSAL_V4_25 full GSM8K completed in `results/causal_v4_25_generalization_v1/gsm8k_full`; smoke outputs remain in the separate `gsm8k` directory.

Completeness audit: 1319/1319 completed, 0 OOM, 0 error, `problem_id` unique. CAUSAL_V4_25 scored 1041/1319 correct = 78.9234% strict accuracy.

## LongBench
Initial CAUSAL_V4_25 3-task x 5 smoke failed due OOM: passage_retrieval_en 5/5 OOM and lcc 1/5 OOM at 8K cap. OOM forensic isolated a system-level prefill logits allocation and cross-sample runtime-state lifecycle issue without changing selector, V4 ratio, sink/recent/residual, group size, quantization semantics, prompt, scorer, or sample subset. After the system-only fix and semantic regression, LONG_BENCH_SMOKE_V2 passed 15/15 with OOM 0 and error 0.

CAUSAL_V4_25 full LongBench 21x50 8K completed in `results/causal_v4_25_generalization_v1/longbench_full`. Completeness audit: 1050/1050 completed, 0 OOM, 0 error, `sample_id` unique. Macro average score: 42.4662.

During final offline completion, multiple sample shards wrote into the same task jsonl. A system-only runner fix added `fcntl.flock` around atomic jsonl rewrites and PID-specific temp files in `scripts/run_longbench_paper_8k_single4090.py::atomic_append_jsonl`; no model, selector, quantization, prompt, scorer, or sample protocol was changed.

## Cross-Benchmark Summary
| Benchmark | FP16 | KIVI | PatternKV | CAUSAL | Δ vs PatternKV | Δ vs FP16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K full strict accuracy | 78.0136 | 68.9158 | 73.7680 | 78.9234 | +5.1554 | +0.9098 |
| LongBench 21x50 8K macro | 43.2862 | 41.2143 | 41.6119 | 42.4662 | +0.8543 | -0.8200 |

## Interpretation
The frozen CAUSAL_V4_25 method completed both generalization targets. It improves over PatternKV and KIVI on GSM8K and LongBench under the reused baseline protocol. It slightly exceeds FP16 on GSM8K but remains below FP16 on LongBench macro average.

## Limitations
CAUSAL uses mixed V2/V4 value precision, so its average formal bit budget is higher than pure K2V2 KIVI/PatternKV baselines. Comparisons are full-method comparisons, not isolated selector benefits. A same-bit RANDOM_V4_25 LongBench/GSM8K control remains recommended; AIME24 has RANDOM_V4_25 artifacts, but no protocol-matched GSM8K/LongBench full results were found in this run.
