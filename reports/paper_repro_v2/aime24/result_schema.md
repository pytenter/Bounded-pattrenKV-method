# AIME24 Result Schema

Each task writes one atomic JSON file:

```text
results/paper_repro_v2/aime24_budget_n2/{method}/p{problem_id:02d}_s{sample_id}_{config_hash}.json
```

Required fields include:

- `experiment_id`
- `dataset`
- `model_path`
- `model_name`
- `method`
- `problem_id`
- `sample_id`
- `task_key`
- `seed`
- `base_seed`
- `config_hash`
- `problem`
- `reference_answer`
- `rendered_prompt`
- `prompt_protocol`
- `chat_template_used`
- `force_think_prefix`
- `input_tokens`
- `do_sample`
- `temperature`
- `top_p`
- `max_new_tokens`
- `generated_text`
- `generated_tokens`
- `total_sequence_tokens`
- `parsed_answer`
- `parser_strategy`
- `parser_error`
- `boxed_candidates`
- `is_correct`
- `stop_reason`
- `hit_max_new_tokens`
- `wall_time_seconds`
- `tokens_per_second`
- `gpu_id`
- `gpu_name`
- `peak_memory_allocated_bytes`
- `peak_memory_reserved_bytes`
- `quantization_config`
- `patternkv_config`
- `cache_bitwidth_stats`
- `git_commit`
- `timestamp`
- `error`

OOM records preserve known context and use `stop_reason=oom`.
