# Effective Bit Accounting

Canonical source: `releases/causal_v4_25_aime24_v1/bit_accounting.json`.

Formal project metric:

- Pattern Base: `2.25` bit/KV element.
- Random-25%: `2.50048828125` bit/KV element.
- CAUSAL-V4@25%: `2.50048828125` bit/KV element.
- Same-bit control valid: `True`.

Scope: payload-and-metadata effective quantization budget. This is not physical Python tensor storage, allocator memory, sink/recent full precision storage, or whole-GPU memory.
