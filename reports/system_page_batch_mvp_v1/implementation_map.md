# Implementation Map

| file | symbol | current B=1 assumption | required B>1 modification | algorithm-semantic risk |
| --- | --- | --- | --- | --- |
| `models/segmented_cache.py` | `_cat_mixed_packed_v` | legacy cache writer rejects `B!=1` and splits with `precision_mask[0]` | keep legacy B1 path; use `quant.page_batch.pack_mixed_v_pages` for page-centric batch ABI | low if selector outputs are consumed unchanged |
| `quant/matmul.py` | `_cuda_attn_v_mixed_fused_with_base_impl` | legacy fused entry rejects `B!=1` | keep legacy B1 reference; use `quant.page_batch.patternkv_page_batch_decode` for standalone page batch MVP | low for correctness, high for performance until CUDA kernel replaces Torch page loop |
| `quant/page_batch.py` | `PatternKVBatchMetadata` | none; request/page metadata is explicit | future ragged extension fills request tables from scheduler/allocator | low |
| `quant/page_batch.py` | `patternkv_page_batch_decode` | fixed-length B in `{1,2,4}` | replace page-local Torch expansion with CUDA/Triton page kernel | low if independent affine streams remain separate |
