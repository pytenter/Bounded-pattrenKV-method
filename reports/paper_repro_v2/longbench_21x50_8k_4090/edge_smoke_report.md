# Edge Smoke Report

Edge smoke reuses a real LongBench sample near the 8K cap. No padding or synthetic text is used.

| method | task | input_tokens | max_gen | stop_reason | peak_reserved_GiB |
| --- | --- | ---: | ---: | --- | ---: |
| fp16 | trec | 8003 | 64 | length | 23.12 |
| kivi_paper_g128 | trec | 8003 | 64 | length | 22.50 |
| patternkv_paper | trec | 8003 | 64 | length | 22.65 |
