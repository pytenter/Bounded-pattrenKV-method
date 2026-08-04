
# Numerical Parity Report

Independent tensor parity test:

- shape: `B=1, Hq=4, Hkv=2, groups=2, q_len=2, kv_len=5, D=8`
- reference: explicit repeat K/V then standard torch QK softmax AV
- fixed path: `kivi_gqa_attention_reference`, which uses the same GQA helper used by KIVI residual paths
- result: `torch.testing.assert_close(..., rtol=1e-4, atol=1e-4)` PASS

Metrics for the independent path:

- max_abs_error: `0.0`
- mean_abs_error: `0.0`
- relative_l2_error: `0.0`

Model-level limitation:

- The official KIVI quantized CUDA kernel does not expose dequantized temporary K/V or per-layer attention output without invasive hooks.
- A real model boundary trace was run for `max_new_tokens=129`; it confirms layer0 persistent cache remains at 8 KV heads while quantized and residual cache coexist after the 128-token boundary.
