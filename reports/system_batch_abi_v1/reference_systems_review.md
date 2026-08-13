# Reference Systems Review

Detailed notes:

- [vLLM TurboQuant](vllm_turboquant_notes.md)
- [SGLang Backend](sglang_backend_notes.md)
- [FlashInfer](flashinfer_design_notes.md)
- [Kitty](kitty_design_notes.md)

## Summary

| System | Directly reusable | Conceptually reusable | Not compatible |
|---|---|---|---|
| vLLM TurboQuant | KV quant mode/backend separation, block tables, seq lengths | backend-specific cache spec and packed decode operator | TurboQuant quantization semantics and single packed slot layout |
| SGLang | attention backend boundary, `ForwardBatch` request metadata, decode-vs-extend split | CUDA graph metadata discipline, KV pool ownership | ordinary FP/FP8 KV pool layout without PatternKV-specific streams |
| FlashInfer | paged metadata shapes (`indptr`, `indices`, `last_page_len`) | plan/run split, page-table thinking, split-KV scheduling | existing kernels cannot restore Pattern centroids or independent V2/V4 affine streams |
| Kitty | none at algorithm level | page-centric mixed-precision layout, avoiding scattered access/divergence | residual/enhancement precision boost and channel-wise boost semantics |

## Direct Kernel Reuse

`DIRECT_KERNEL_REUSE=NO` for FlashInfer and other existing attention kernels. PatternKV decode requires:

- K INT2 residual with Pattern centroid compensation;
- V2 and V4 independent affine payloads;
- logical precision bitmap/rank metadata;
- compact Pattern V gate and centroid assignment restoration.

Existing kernels do not expose this combined ABI.

## Metadata/Scheduling Reuse

`REUSE_METADATA_AND_SCHEDULING_DESIGN=YES`.

The serving systems agree on one important lesson: batch decode should be represented by request/page metadata, not dense `[B,H,T,D]` tensors.
