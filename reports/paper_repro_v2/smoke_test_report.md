# Paper Repro v2 Smoke Test Report

Status: SMOKE COMPAT PASS

## 31500-token Paper Smoke Attempt

Result: PARTIAL. qasper succeeded, but passage_retrieval_en/lcc OOM at max_input_length=31500 on RTX 3090 24GB in this implementation. Failed records were preserved under `results/paper_repro_v2/smoke`.

```json
[
  {
    "method": "fp16",
    "task": "passage_retrieval_en",
    "error": "OutOfMemoryError('CUDA out of memory. Tried to allocate 5.36 GiB. GPU 0 has a total capacity of 23.56 GiB of which 3.77 GiB is free. Including non-PyTorch memory, this process has 19.77 GiB memory in use. Of the allocated memory 19.10 GiB i"
  },
  {
    "method": "fp16",
    "task": "lcc",
    "error": "OutOfMemoryError('CUDA out of memory. Tried to allocate 5.46 GiB. GPU 0 has a total capacity of 23.56 GiB of which 3.68 GiB is free. Including non-PyTorch memory, this process has 19.85 GiB memory in use. Of the allocated memory 19.18 GiB i"
  },
  {
    "method": "kivi_paper_g128",
    "task": "passage_retrieval_en",
    "error": "OutOfMemoryError('CUDA out of memory. Tried to allocate 5.36 GiB. GPU 0 has a total capacity of 23.56 GiB of which 4.83 GiB is free. Including non-PyTorch memory, this process has 18.71 GiB memory in use. Of the allocated memory 17.93 GiB i"
  },
  {
    "method": "kivi_paper_g128",
    "task": "lcc",
    "error": "OutOfMemoryError('CUDA out of memory. Tried to allocate 5.46 GiB. GPU 0 has a total capacity of 23.56 GiB of which 4.75 GiB is free. Including non-PyTorch memory, this process has 18.79 GiB memory in use. Of the allocated memory 17.99 GiB i"
  },
  {
    "method": "patternkv_paper",
    "task": "passage_retrieval_en",
    "error": "OutOfMemoryError('CUDA out of memory. Tried to allocate 5.36 GiB. GPU 0 has a total capacity of 23.56 GiB of which 4.78 GiB is free. Including non-PyTorch memory, this process has 18.76 GiB memory in use. Of the allocated memory 17.98 GiB i"
  },
  {
    "method": "patternkv_paper",
    "task": "lcc",
    "error": "OutOfMemoryError('CUDA out of memory. Tried to allocate 5.46 GiB. GPU 0 has a total capacity of 23.56 GiB of which 4.21 GiB is free. Including non-PyTorch memory, this process has 19.32 GiB memory in use. Of the allocated memory 18.04 GiB i"
  }
]
```

## 8192 Compatibility Smoke

This verifies the paper method aliases, group size, axis semantics, output writing, and cache bitwidth fields without overwriting the failed 31500-token attempt.

| method | task | rows | errors | empty | score | input | output | truncated | theoretical quant bits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| fp16 | qasper | 1 | 0 | 0 | 29.41 | 4034 | 34 | False | 16.0 |
| fp16 | passage_retrieval_en | 1 | 0 | 0 | 0.0 | 8183 | 32 | True | 16.0 |
| fp16 | lcc | 1 | 0 | 0 | 100.0 | 8168 | 64 | True | 16.0 |
| kivi_paper_g128 | qasper | 1 | 0 | 0 | 25.0 | 4034 | 42 | False | 2.25 |
| kivi_paper_g128 | passage_retrieval_en | 1 | 0 | 0 | 0.0 | 8183 | 32 | True | 2.25 |
| kivi_paper_g128 | lcc | 1 | 0 | 0 | 100.0 | 8168 | 64 | True | 2.25 |
| patternkv_paper | qasper | 1 | 0 | 0 | 26.67 | 4034 | 30 | False | 2.25 |
| patternkv_paper | passage_retrieval_en | 1 | 0 | 0 | 0.0 | 8183 | 32 | True | 2.25 |
| patternkv_paper | lcc | 1 | 0 | 0 | 100.0 | 8168 | 64 | True | 2.25 |

## Boundary Check

The unit test `tests/test_paper_v2_config.py` asserts no dynamic PatternKV event before token 128 and one event at token 128/129 using the configured residual window. The real smoke outputs are at most 64 tokens, so they do not naturally cross a second residual-window boundary.

## Issues

No integrity issues in the 8192 compatibility smoke.
