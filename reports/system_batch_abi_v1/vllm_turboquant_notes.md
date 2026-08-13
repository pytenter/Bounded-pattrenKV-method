# vLLM TurboQuant Notes

Sources:

- vLLM `kv_cache_interface.py`: https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py
- vLLM `turboquant_attn.py`: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/turboquant_attn.py
- vLLM `triton_turboquant_decode.py`: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_turboquant_decode.py
- vLLM `turboquant_soa/`: https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/ops/turboquant_soa
- vLLM cache config: https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py

## Observations

- TurboQuant is represented as a distinct KV quantization mode: `KVQuantMode.TURBOQUANT`.
- The KV cache spec layer budgets cache pages by `block_size`, number of KV heads, head size, dtype, and quant mode; it does not force all quant modes into a plain FP16 KV layout.
- TurboQuant decode takes a compact quantized `kv_cache` plus `block_table` and `seq_lens`. The public wrapper signature in `triton_turboquant_decode.py` uses:
  - `query: [B,Hq,D]`
  - `kv_cache: [num_blocks, block_size, Hk, padded_slot] uint8`
  - `block_table: [B,max_num_blocks] int32`
  - `seq_lens: [B] int32`
  - quant parameters such as `Pi`, `centroids`, `mse_bits`, key packed size, and value bits.
- The Triton decode grid is `(B,Hq,NUM_KV_SPLITS)`, so batch and split-KV scheduling are explicit kernel axes.
- `block_table` maps request-local block id to physical cache block id; `seq_lens` carries ragged request lengths.
- The quantized store and decode operator are separated: the cache stores packed bytes in a TurboQuant-specific layout, while the backend/operator interprets that layout.
- Backend registration is done as an attention backend path, not by pretending TurboQuant is normal FP16 KV.

## Answers

- Dedicated KV Cache Spec? Yes, via `KVQuantMode.TURBOQUANT` and attention spec/cache dtype plumbing. It is a quant mode with a backend-specific packed layout.
- Cache slot representation? A physical block/page id plus an offset inside a fixed-size block. The packed slot is uint8 and backend-specific.
- Block table representation? Per request, dense table `[B,max_num_blocks]` mapping logical blocks to physical cache blocks.
- `slot_mapping` use? vLLM generally uses slot mappings for writes/appends into physical KV slots; TurboQuant decode itself uses block tables and seq lengths for reads.
- Batch/ragged metadata to kernel? `block_table`, `seq_lens`, fixed `block_size`, and split-KV parameters.
- Quantized KV store vs decode separation? Yes. Storage ABI is quantized and page/block based; decode knows how to dequantize/accumulate.
- Backend registration? Through the attention backend and KV quant mode dispatch.
- Why not normal FP16 layout? The compressed cache has different bytes/slot, extra quant metadata/rotations/centroids, and decode-specific dequantization. Forcing FP16 layout would destroy the memory win and require materialization.

## PatternKV Reuse

Directly reusable:

- block table + sequence length contract;
- backend-specific KV cache spec;
- separating append/write layout from decode operator.

Conceptually reusable:

- packed uint8 page slot design;
- split-KV grid `(batch, head, kv_split)`;
- backend-owned workspace and metadata.

Not directly reusable:

- TurboQuant's Hadamard/Lloyd-Max quantization semantics;
- a single packed K+V slot if PatternKV keeps asymmetric K tight and V page/capacity-friendly.
