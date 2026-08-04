# GSM8K Smoke Truncation Analysis

- root: `results/gsm8k/smoke_1024_existing_mixed_gpu`

## fp16

- summary: `{'rows': 50, 'correct': 41, 'accuracy': 82.0, 'length_truncated': 1, 'parser_failures': 0, 'errors': 0, 'avg_output_tokens': 274.56, 'median_output_tokens': 237.5, 'p95_output_tokens': 409, 'max_output_tokens': 1024, 'normal_eos_rate': 0.0}`
- reached_limit_or_truncated_rows: `1`

## kivi

- summary: `{'rows': 50, 'correct': 18, 'accuracy': 36.0, 'length_truncated': 20, 'parser_failures': 0, 'errors': 0, 'avg_output_tokens': 590.88, 'median_output_tokens': 357.0, 'p95_output_tokens': 1024, 'max_output_tokens': 1024, 'normal_eos_rate': 0.0}`
- reached_limit_or_truncated_rows: `20`
- kivi_truncated_classes: `{'E_repetition_or_loop': 17, 'F_other': 3}`
- kivi_boxed_stats: `{'boxed_count': 0, 'boxed_correct_count': 0, 'boxed_wrong_count': 0, 'no_boxed_count': 20, 'loop_or_repetition_count': 17, 'parser_failure_count': 0}`

## patternkv

- summary: `{'rows': 50, 'correct': 42, 'accuracy': 84.0, 'length_truncated': 1, 'parser_failures': 0, 'errors': 0, 'avg_output_tokens': 259.8, 'median_output_tokens': 238.5, 'p95_output_tokens': 364, 'max_output_tokens': 1024, 'normal_eos_rate': 0.0}`
- reached_limit_or_truncated_rows: `1`
