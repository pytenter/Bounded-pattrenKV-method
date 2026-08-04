# GSM8K ModelScope Dataset Report

Generated at: 2026-08-03T19:43:34.862317
Dataset ID: modelscope/gsm8k
Load attempt: `{'fallback': '/data/zypan/.cache/modelscope/hub/datasets/swift___gsm8k/default-db951f94ee73f5bc/0.0.0/master/gsm8k-test.arrow', 'last_ms_dataset_error': 'TypeError("DatasetBuilder.as_dataset() got an unexpected keyword argument \'verification_mode\'")'}`
Revision: default
Split: test
Samples: 1319
Local path: data/gsm8k/test.jsonl
Hugging Face access: not used by this script

## First Three Field Structures

```json
[
  {
    "sample_index": "int",
    "question": "str",
    "answer": "str",
    "gold_answer": "str"
  },
  {
    "sample_index": "int",
    "question": "str",
    "answer": "str",
    "gold_answer": "str"
  },
  {
    "sample_index": "int",
    "question": "str",
    "answer": "str",
    "gold_answer": "str"
  }
]
```

## Validation

PASS: 1319 samples, unique sample_index, non-empty question/answer, parseable gold_answer.
