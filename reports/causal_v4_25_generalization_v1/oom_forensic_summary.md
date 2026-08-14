# LongBench OOM Forensic Summary

## Samples

| role | task | sample id | raw tokens | tokenized tokens | max new |
| --- | --- | --- | ---: | ---: | ---: |
| stable OOM | passage_retrieval_en | `passage_retrieval_en:f440b317fe84ec87132b316031ad47925a6ae33e0ef0ee3a` | 11176 | 8183 | 32 |
| stable OOM | lcc | `lcc:28ce5cf5e0beeb47248c6cba20ae17fa71d60d402770d464` | 11436 | 8168 | 64 |
| control | qasper | `qasper:3ac3eef636db11635a21a61804cb28e92c546a5686dd1e12` | 3999 | 4034 | 128 |

## Diagnostic Memory

All values are bytes. Diagnostic runs use the same frozen CAUSAL_V4_25 config and same problem ids; PatternKV runs are memory diagnostics only and do not replace historical benchmark results.

| task | method | max new | OOM_STAGE | generated | prefill peak allocated | prefill peak reserved | decode peak allocated | decode peak reserved |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| passage_retrieval_en | CAUSAL_V4_25 | 1 | None | 1 | 18978441216 | 19805503488 | 17607760896 | 18960351232 |
| passage_retrieval_en | CAUSAL_V4_25 | 32 | None | 32 | 18978441216 | 19805503488 | 17911492096 | 19092471808 |
| passage_retrieval_en | PatternKV diagnostic | 32 | None | 32 | 18840470016 | 19715325952 | 17649350656 | 18912116736 |
| lcc | CAUSAL_V4_25 | 64 | None | 64 | 18977441792 | 19803406336 | 17915850752 | 19090374656 |
| lcc | PatternKV diagnostic | 64 | None | 64 | 18833339904 | 19713228800 | 17652348928 | 18914213888 |
| qasper | CAUSAL_V4_25 | 128 | None | 128 | 20526707712 | 21988638720 | 17576095232 | 18517852160 |
| qasper | PatternKV diagnostic | 128 | None | 128 | 17991553024 | 18589155328 | 17457325056 | 18178113536 |

## Root Cause

The OOM was not caused by the frozen selector, V4 ratio, group size, sink/recent/residual, prompt, scorer, or sample subset. The immediate allocation cliff was in `LlamaForCausalLM_PatternKV.forward`: during prefill under `generate`, it materialized full sequence vocabulary logits even though generation only consumes the final prefill token. A second system issue was sample lifecycle leakage: PatternKV runtime centroid state was not reset when moving between CAUSAL samples.

## Fix

The fix is system-only:
- project only the last hidden state for inference prefill logits in `models/llama_patternkv.py`;
- reset PatternKV runtime state in CAUSAL sample context setup for GSM8K and LongBench runners.

Semantic regression passed on the qasper control sample: V4 indices exact, V4 count exact, quantization config exact, cache hash exact, next-token argmax identical. Existing AIME frozen/original tests passed: `42 passed`.
