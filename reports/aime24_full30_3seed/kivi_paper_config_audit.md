# KIVI Paper Configuration Audit

- Config name: `kivi_paper`
- Backend method: `kivi_official`
- Cache mode: `legacy_tuple_chunked`
- PatternKV cache path: `segmented`
- PatternKV cache mode: `segmented_rolling`
- Residual length: `128`
- Sink length: `0`
- Recent length: `128`
- K bits: `2`
- V bits: `2`
- Group size: `128`
- K quantization granularity: `per-channel: quantize transposed K along token axis`
- V quantization granularity: `per-token: quantize V along head_dim axis`
- Asymmetric quantization: `True`
- Quantized-region affine bits: `2.25`
- Metadata: packed K/V payload plus FP16 scale/min metadata.
- Cache semantics: KIVI official chunk residual semantics with residual_length=128 and no Sink protection.
