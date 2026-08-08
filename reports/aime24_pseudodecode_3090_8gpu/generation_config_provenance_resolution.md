# Generation Config Provenance Resolution

## 1. Why `a7d6...` Exists

The frozen hash comes from `reports/aime24_int2_wave1_v100_8gpu/revised_wave1a_full_run_manifest.json` at source commit `232e3b08d10919ca24932ad0a0135e46119ecfd5`.

## 2. Exact Historical Payload

```json
{
  "batch_size": 1,
  "configs": [
    "pattern_legacy_chunked_k2v2_r128",
    "pattern_rolling_k2v2_s0_r128",
    "pattern_rolling_k2v2_s64_r256",
    "pattern_rolling_k4v2_s0_r128",
    "pattern_rolling_k2v4_s0_r128",
    "kivi_legacy_chunked_k2v2_r128",
    "kivi_rolling_k2v2_s0_r128",
    "kivi_rolling_k2v2_s64_r256"
  ],
  "do_sample": true,
  "dtype": "float16",
  "manifest_hash": "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e",
  "max_model_len": 131072,
  "max_new_tokens": 32768,
  "model": "/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B",
  "seed": 42,
  "task_count": 12,
  "temperature": 0.6,
  "top_p": 0.95
}
```

## 3. Canonical Serialization Rule

`json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`, then SHA256 truncated to 32 hex characters.

## 4. Proof That `a7d6...` Can Be Reproduced

- Legacy expected hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
- Legacy recomputed hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
- HISTORICAL_GENERATION_HASH_REPRODUCED: `True`

## 5. Why `de91...` Differs

`de91...` is the hash of the current helper schema, `bench.aime_utils.generation_config_dict`. That schema has different fields from the legacy run manifest schema, so equality to `a7d6...` is not the right test.

- Current helper hash: `de91b2ba80450d53c10210854f265abd`
- GENERATION_HASH_SCHEMA_MISMATCH_CONFIRMED: `True`

## 6. Why The Legacy Hash Is Nonportable

The legacy payload includes the V100 server absolute model path. The 3090 server has a different local model path, so path-inclusive hash equality would reject a semantically compatible independent server.

- LEGACY_HASH_PORTABLE: `False`
- Legacy nonportable fields: `['model absolute path']`

## 7. Which Semantics Affect Reference Trajectories

The portable reference-generation fingerprint includes task cohort, task seeds, model/tokenizer identity, dtype, prompt construction, chat template semantics, sampling controls, EOS/PAD resolution, max-new-token limit, and context compatibility. It excludes machine-local paths and quantized method sets.

## 8. New Portable Schema

- Schema version: `aime24_reference_generation_semantics_v1`

```json
{
  "add_special_tokens": false,
  "base_seed": 42,
  "chat_template": "tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
  "context_limit": 131072,
  "dataset": "aime24",
  "do_sample": true,
  "force_think_prefix": true,
  "max_new_tokens": 32768,
  "model_dtype": "float16",
  "model_identity_hash": "54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd",
  "model_name": "DeepSeek-R1-Distill-Llama-8B",
  "num_return_sequences": 1,
  "ordered_task_identity": [
    {
      "problem_id": 12,
      "sample_id": 0,
      "seed": 12042,
      "task_key": "aime24:p12:s0:seed12042"
    },
    {
      "problem_id": 14,
      "sample_id": 0,
      "seed": 14042,
      "task_key": "aime24:p14:s0:seed14042"
    },
    {
      "problem_id": 15,
      "sample_id": 1,
      "seed": 15043,
      "task_key": "aime24:p15:s1:seed15043"
    },
    {
      "problem_id": 16,
      "sample_id": 0,
      "seed": 16042,
      "task_key": "aime24:p16:s0:seed16042"
    },
    {
      "problem_id": 11,
      "sample_id": 0,
      "seed": 11042,
      "task_key": "aime24:p11:s0:seed11042"
    },
    {
      "problem_id": 12,
      "sample_id": 1,
      "seed": 12043,
      "task_key": "aime24:p12:s1:seed12043"
    },
    {
      "problem_id": 6,
      "sample_id": 0,
      "seed": 6042,
      "task_key": "aime24:p6:s0:seed6042"
    },
    {
      "problem_id": 6,
      "sample_id": 1,
      "seed": 6043,
      "task_key": "aime24:p6:s1:seed6043"
    },
    {
      "problem_id": 0,
      "sample_id": 0,
      "seed": 42,
      "task_key": "aime24:p0:s0:seed42"
    },
    {
      "problem_id": 11,
      "sample_id": 1,
      "seed": 11043,
      "task_key": "aime24:p11:s1:seed11043"
    },
    {
      "problem_id": 15,
      "sample_id": 0,
      "seed": 15042,
      "task_key": "aime24:p15:s0:seed15042"
    },
    {
      "problem_id": 17,
      "sample_id": 0,
      "seed": 17042,
      "task_key": "aime24:p17:s0:seed17042"
    }
  ],
  "prompt_protocol": "deepseek_r1_recommended",
  "repetition_penalty": 1.0,
  "resolved_eos_token_ids": [
    128001,
    128009
  ],
  "resolved_pad_token_id": 128001,
  "schema_version": "aime24_reference_generation_semantics_v1",
  "task_count": 12,
  "task_manifest_sha256": "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e",
  "task_seed_algorithm": "effective_seed = base_seed + problem_id * 1000 + sample_id",
  "task_seed_map": [
    [
      "aime24:p12:s0:seed12042",
      12042
    ],
    [
      "aime24:p14:s0:seed14042",
      14042
    ],
    [
      "aime24:p15:s1:seed15043",
      15043
    ],
    [
      "aime24:p16:s0:seed16042",
      16042
    ],
    [
      "aime24:p11:s0:seed11042",
      11042
    ],
    [
      "aime24:p12:s1:seed12043",
      12043
    ],
    [
      "aime24:p6:s0:seed6042",
      6042
    ],
    [
      "aime24:p6:s1:seed6043",
      6043
    ],
    [
      "aime24:p0:s0:seed42",
      42
    ],
    [
      "aime24:p11:s1:seed11043",
      11043
    ],
    [
      "aime24:p15:s0:seed15042",
      15042
    ],
    [
      "aime24:p17:s0:seed17042",
      17042
    ]
  ],
  "temperature": 0.6,
  "think_prefix": "<think>\n",
  "tokenizer_identity_hash": "b9c9eb63a8e03059914880f918cd28a880dec8b6e15e4461e1ff677e3743dbb8",
  "top_p": 0.95,
  "user_prompt_template": "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{}."
}
```

## 9. New Portable Hash

- Portable reference generation hash: `86648d12304ce11890c1a8f64bf5a896`
- Experiment config set hash: `721d12131e83f6fab91368169c86fce8`

## 10. Final Gate Decision

- TOKENIZER_IDENTITY_VALID: `True`
- CONTEXT_SEMANTICS_VALID: `True`
- PORTABLE_GENERATION_SEMANTICS_VALID: `True`
- PORTABLE_PROMPT_PIPELINE_VALID: `True`
- GENERATION_CONFIG_VALID: `True`

Formal run remains blocked by later preflight gates such as FP16 zero-gap, static independence, pseudo feedback, production parity, and observer non-invasiveness.
