
# GSM8K Smoke Report

Result directory: `results/paper_repro_v2/gsm8k_paper_smoke_gqa_fixed`

| 方法 | planned | completed | error | OOM | persistent KV heads | 配置正确 |
|---|---:|---:|---:|---:|---:|---|
| FP16 | 3 | 3 | 0 | 0 | 8 | YES |
| KIVI G128 | 3 | 3 | 0 | 0 | 8 | YES |
| PatternKV | 3 | 3 | 0 | 0 | 8 | YES |

Notes:

- All three methods used problem IDs `0,1,2`, greedy decoding, batch size 1, and `max_new_tokens=1024`.
- KIVI result config remains INT2 G128/R128 with K per-channel and V per-token quantization.
- No GSM8K full or LongBench full run was started in this round.

## Boundary Trace

Layer0 direct decode trace from `logs/paper_repro_v2/kivi_gqa_fix/kivi_boundary_cache_trace.log`:

```json
{
  "max_memory_allocated": 16165611520,
  "max_memory_reserved": 16338911232,
  "records": [
    {
      "generated_step": 0,
      "key_full_shape": [
        1,
        8,
        106,
        128
      ],
      "key_persistent_bytes": 217088,
      "key_persistent_heads": 8,
      "key_quant_trans_shape": null,
      "kv_len": 106,
      "value_full_shape": [
        1,
        8,
        106,
        128
      ],
      "value_persistent_bytes": 217088,
      "value_persistent_heads": 8,
      "value_quant_shape": null
    },
    {
      "generated_step": 1,
      "key_full_shape": [
        1,
        8,
        107,
        128
      ],
      "key_persistent_bytes": 219136,
      "key_persistent_heads": 8,
      "key_quant_trans_shape": null,
      "kv_len": 107,
      "value_full_shape": [
        1,
        8,
        107,
        128
      ],
      "value_persistent_bytes": 219136,
      "value_persistent_heads": 8,
      "value_quant_shape": null
    },
    {
      "generated_step": 22,
      "key_full_shape": null,
      "key_persistent_bytes": 36864,
      "key_persistent_heads": 8,
      "key_quant_trans_shape": [
        1,
        8,
        128,
        8
      ],
      "kv_len": 128,
      "value_full_shape": [
        1,
        8,
        128,
        128
      ],
      "value_persistent_bytes": 262144,
      "value_persistent_heads": 8,
      "value_quant_shape": null
    },
    {
      "generated_step": 23,
      "key_full_shape": [
        1,
        8,
        1,
        128
      ],
      "key_persistent_bytes": 38912,
      "key_persistent_heads": 8,
      "key_quant_trans_shape": [
        1,
        8,
        128,
        8
      ],
      "kv_len": 129,
      "value_full_shape": [
        1,
        8,
        128,
        128
      ],
      "value_persistent_bytes": 262432,
      "value_persistent_heads": 8,
      "value_quant_shape": [
        1,
        8,
        1,
        8
      ]
    },
    {
      "generated_step": 24,
      "key_full_shape": [
        1,
        8,
        2,
        128
      ],
      "key_persistent_bytes": 40960,
      "key_persistent_heads": 8,
      "key_quant_trans_shape": [
        1,
        8,
        128,
        8
      ],
      "kv_len": 130,
      "value_full_shape": [
        1,
        8,
        128,
        128
      ],
      "value_persistent_bytes": 262720,
      "value_persistent_heads": 8,
      "value_quant_shape": [
        1,
        8,
        2,
        8
      ]
    },
    {
      "generated_step": 129,
      "key_full_shape": [
        1,
        8,
        107,
        128
      ],
      "key_persistent_bytes": 256000,
      "key_persistent_heads": 8,
      "key_quant_trans_shape": [
        1,
        8,
        128,
        8
      ],
      "kv_len": 235,
      "value_full_shape": [
        1,
        8,
        128,
        128
      ],
      "value_persistent_bytes": 292960,
      "value_persistent_heads": 8,
      "value_quant_shape": [
        1,
        8,
        107,
        8
      ]
    }
  ]
}
```
