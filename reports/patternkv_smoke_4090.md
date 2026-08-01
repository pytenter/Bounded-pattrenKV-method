# PatternKV Smoke Report

Requested target: single RTX 4090 24GB / SM89
Actual GPU: NVIDIA GeForce RTX 3090 capability [8, 6]

## Smoke Results

| method | input tokens | output tokens | error | OOM | latency s | cuda ms | peak reserved GB |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| fp16 | 296 | 160 | None | False | 7.072 | 7071.5 | 16.54 |
| patternkv | 296 | 160 | None | False | 9.703 | 9703.1 | 16.53 |

## PatternKV Path Coverage

- Prefill input length exceeded residual_length: YES
- Historical K cache packed int32 exists: YES
- Historical V cache packed int32 exists: YES
- FP16 residual window retained: YES
- K pattern index generated: YES
- V mask generated as uint8: YES
- Decode pattern update occurred: YES
- CUDA fused attention path callable: YES
- No full FP16 historical KV fallback in cache: YES

## Layer 0 Tensor Evidence
```json
{
  "layer_stats": {
    "layer": 0,
    "k_base": {
      "shape": [
        8,
        33,
        128
      ],
      "dtype": "torch.float16"
    },
    "v_centroids": {
      "shape": [
        8,
        33,
        128
      ],
      "dtype": "torch.float16"
    }
  },
  "cache": {
    "layer": 0,
    "key_states_quant_trans": {
      "shape": [
        1,
        8,
        128,
        24
      ],
      "dtype": "torch.int32"
    },
    "key_states_full": {
      "shape": [
        1,
        8,
        71,
        128
      ],
      "dtype": "torch.float16"
    },
    "key_scale_trans": {
      "shape": [
        1,
        8,
        128,
        3
      ],
      "dtype": "torch.float16"
    },
    "key_mn_trans": {
      "shape": [
        1,
        8,
        128,
        3
      ],
      "dtype": "torch.float16"
    },
    "value_states_quant": {
      "shape": [
        1,
        8,
        384,
        8
      ],
      "dtype": "torch.int32"
    },
    "value_states_full": {
      "shape": [
        1,
        8,
        71,
        128
      ],
      "dtype": "torch.float16"
    },
    "value_scale": {
      "shape": [
        1,
        8,
        384,
        1
      ],
      "dtype": "torch.float16"
    },
    "value_mn": {
      "shape": [
        1,
        8,
        384,
        1
      ],
      "dtype": "torch.float16"
    },
    "kv_seq_len": 455,
    "k_assignments": {
      "shape": [
        1,
        8,
        384
      ],
      "dtype": "torch.int64",
      "min": 0,
      "max": 32
    },
    "v_mask": {
      "shape": [
        1,
        8,
        384
      ],
      "dtype": "torch.uint8",
      "mean": 0.9762369990348816
    },
    "v_assignments_idx": {
      "shape": [
        1,
        8,
        384
      ],
      "dtype": "torch.int64",
      "min": 0,
      "max": 32
    }
  }
}
```
