# Component Schema

- `qk_int2_history`: packed historical INT2 K score production, including centroid-base compensated CUDA QK reader.
- `qk_fp16_sink`, `qk_fp16_pending`, `qk_fp16_recent`: FP16 K segment score production through request-invariant elementwise multiply/sum.
- `attention_score_concat`: physical segment score concatenation into one `[B,H,1,T]` logits tensor.
- `attention_softmax`: request-invariant fixed-split softmax over concatenated physical logits; internal max/sum state is not exposed.
- `mixed_v_page_pool_operator`: fused paged historical mixed V2/V4 Value operator.
- `value_fp16_tail`: aggregate of three FP16 Value segment calls: sink, pending, recent. It excludes historical V2/V4.
- `cache_append`: per-layer decode K/V append and recent/pending state mutation.
- `page_batch_pack` and `pack_window`: boundary maintenance ranges; expected to be zero in decode=8 because `group_size=128`.

Profile ranges are nested. Percentages are attribution aids, not additive closure.
