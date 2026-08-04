
# KIVI GQA Error Reproduction

- branch: `repro/patternkv-paper-longbench-gsm8k-rerun`
- commit: `4c6ac3b5df971f5d8e1ec636dffd3793aa78e6bd`
- command: `CUDA_VISIBLE_DEVICES=0 python reproduce_kivi_gqa_error problem_id=0 max_new_tokens=64 batch_size=1`
- log: `logs/paper_repro_v2/kivi_gqa_fix/reproduce_error.log`

## Environment

- torch: `2.4.1+cu124`
- transformers: `4.43.1`
- torch CUDA: `12.4`
- GPU: `NVIDIA GeForce RTX 3090`, capability `(8, 6)`
- model: `/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct`
- model heads: `num_attention_heads=32`, `num_key_value_heads=8`, `num_key_value_groups=4`, `head_dim=128`

## Root Failure

The first failing repository frame is the original `models/llama_kivi.py:380` in `LlamaFlashAttention_KIVI.forward`:

```text
attn_output = torch.matmul(attn_weights, value_states_full)
RuntimeError: The size of tensor a (32) must match the size of tensor b (8) at non-singleton dimension 1
```

This happens in **decode**, after a successful flash-attention prefill. The failing operation is ordinary PyTorch AV matmul over the FP16 residual value cache, not the custom CUDA quantized matmul kernel.

## Shapes At Failure

From the reproduced prompt and post-fix debug of the same path:

- prefill input: `query=(1,32,106,128)`, `key=(1,8,106,128)`, `value=(1,8,106,128)`
- first decode QK: `query=(1,32,1,128)`, residual K cache after append `(1,8,107,128)`, temporary K for attention `(1,32,107,128)`
- failing original AV branch: `attn_weights=(1,32,1,107)`, residual V cache after append should be `(1,8,107,128)` and must be temporarily repeated to `(1,32,107,128)`
- quantized K/V cache at the first failing token: `None`; residual-only branch was the immediate failure.

## Full Traceback

```text
Traceback (most recent call last):
  File "<stdin>", line 59, in <module>
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/transformers/generation/utils.py", line 1989, in generate
    result = self._sample(
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/transformers/generation/utils.py", line 2932, in _sample
    outputs = self(**model_inputs, return_dict=True)
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/zypan/PatternKV-repro/models/llama_kivi.py", line 862, in forward
    outputs = self.model(
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/zypan/PatternKV-repro/models/llama_kivi.py", line 751, in forward
    layer_outputs = decoder_layer(
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/zypan/PatternKV-repro/models/llama_kivi.py", line 611, in forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/zypan/.local/share/mamba/envs/patternkv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/zypan/PatternKV-repro/models/llama_kivi.py", line 380, in forward
    attn_output = torch.matmul(attn_weights, value_states_full)
RuntimeError: The size of tensor a (32) must match the size of tensor b (8) at non-singleton dimension 1
```
