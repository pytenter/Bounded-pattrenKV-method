# Timer Scope Audit

- Profile-off TPOT is measured separately and is the only real E2E performance number.
- Profile-on component shares are approximate diagnostic shares.
- Parent timers such as `decode_decoder_model_forward` are excluded from `component_breakdown.csv` shares to avoid double-counting.
- `mixed_v` uses the inclusive `mixed_v_fused_attention` range; its nested mapping/V2/V4/reduce timers are reported in specialized CSVs, not added to the top-level share.
- `other` is derived as profile-on decode wall time minus selected top-level groups.

| Component | Scope | Nested Handling |
|---|---|---|
| QK | query x historical K, including Pattern K fused quantized path plus FP16 sink/pending/recent score regions and score concat | top-level exclusive group or derived residual |
| softmax | attention score normalization | top-level exclusive group or derived residual |
| importance_update | causal importance statistics update | top-level exclusive group or derived residual |
| mixed_v | compressed-domain V2/V4 Value attention | top-level exclusive group or derived residual |
| selector | V4 token identity selector | top-level exclusive group or derived residual |
| packing | new historical V2/V4 quantization and packing at flush | top-level exclusive group or derived residual |
| cache_mutation | append, recent, pending, historical tensor mutation and torch.cat cache changes | top-level exclusive group or derived residual |
| output_projection | attention output linear projection | top-level exclusive group or derived residual |
| qkv_projection | Q/K/V linear projections | top-level exclusive group or derived residual |
| rope | rotary position embedding application | top-level exclusive group or derived residual |
| lm_head | decode LM head projection | top-level exclusive group or derived residual |
| other | unclassified model compute, including layernorm/MLP and any uninstrumented gaps | top-level exclusive group or derived residual |
