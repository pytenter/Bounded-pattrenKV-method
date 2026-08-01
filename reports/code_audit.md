# PatternKV Official Code Audit

Repository: https://github.com/HCOOOH/PatternKV.git  
Commit: aba09a8 (`v0.1.0`)  
Audit time: 2026-08-01

## Files Reviewed

- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `example.py`
- `models/llama_patternkv.py`
- `quant/setup.py`
- `quant/matmul.py`
- `quant/new_pack.py`
- `quant/qmodule.py`
- `quant/test.py`
- `quant/csrc/pybind.cpp`
- `quant/csrc/gemv_cuda.cu`

## Answers

1. **Official code supports which model architectures?**
   The executable implementation is for Llama-family decoder models. `README.md` and `example.py` import/use `LlamaConfig` and `LlamaForCausalLM_PatternKV`. Although README imports mention `Qwen2Config` and `MistralConfig` in the snippet, no Qwen2/Mistral PatternKV model classes are present in the repository.

2. **Is only the Llama path truly implemented?**
   Yes. The only model implementation file is `models/llama_patternkv.py`, defining `LlamaAttention_PatternKV`, `LlamaFlashAttention_PatternKV`, `LlamaModel_PatternKV`, and `LlamaForCausalLM_PatternKV`.

3. **Where is Prefill Pattern mining implemented?**
   In `LlamaFlashAttention_PatternKV.forward` when `past_key_value is None`: K prefill mining is lines 887-896 via `batched_kmeans_fast_compiled` and `batched_assign_compiled`; V prefill mining is lines 901-912 via the same helpers.

4. **Where are K and V clustering implemented?**
   Shared PyTorch KMeans helpers are `batched_kmeans_fast` / `batched_kmeans` in `models/llama_patternkv.py` lines 283-431. K calls: lines 887-896. V calls: lines 901-912. The imported `cuml.cluster.KMeans` is not used in the observed call chain.

5. **Where is min-max Pattern assignment implemented?**
   Decode min-max assignment is `_assign_minmax_hnk` lines 481-499 for K and `_nearest_v_centroid` lines 599-611 for V. Prefill assignment uses Euclidean `batched_assign_compiled` lines 452-474, not min-max.

6. **Where is Decode Chebyshev-center update implemented?**
   `_chebyshev_center_per_head` lines 502-508 computes per-head `(max+min)/2`. Decode K appends a new centroid at lines 703-718. Decode V appends via `_append_v_centroid_from_window` lines 511-531 and call site lines 820-825.

7. **Where are V threshold and mask implemented?**
   `_threshold_and_mask_given_base` lines 533-567 computes range contraction ratio and boolean mask. `_v_threshold_and_mask` lines 570-585 wraps it. Prefill V mask call is lines 966-974; decode V mask call is lines 829-835. Stored V mask is converted to `torch.uint8` at lines 846 and 852, prefill lines 982-983.

8. **K residual quantization call path.**
   Prefill: `LlamaFlashAttention_PatternKV.forward` builds K centroids/assignments, gathers bases, subtracts them at lines 931-938, then calls `triton_quantize_and_pack_along_last_dim` at line 939. Decode: lines 703-742 append centroid, assign by min-max, subtract base, and quantize/pack the residual window.

9. **V residual quantization call path.**
   Prefill: V centroids and assignment at lines 901-912, gather base lines 966-967, threshold/mask lines 969-974, then `triton_quantize_and_pack_along_last_dim` lines 977-979. Decode: V window reaches `residual_length`, appends centroid lines 820-825, assigns and masks lines 829-835, subtracts masked base line 835, then quantizes/pack at lines 838-840.

10. **What does the CUDA extension do in attention?**
    `patternkv_gemv` exposes `gemv_forward_cuda_outer_dim`, `gemv_forward_cuda_outer_dim_with_base`, and `attn_v_forward_cuda_outer_dim_with_base` via `quant/csrc/pybind.cpp`. In attention it computes fused query-times-quantized-K with centroid-base compensation (`cuda_bmm_fA_qB_outer_with_base`) and fused attention-times-quantized-V with V centroid/mask compensation (`cuda_attn_v_fused_with_base`).

11. **Triton vs custom CUDA usage.**
    Triton: `quant/new_pack.py` `_minmax_along_last_dim` and `_pack_along_last_dim` implement min/max and pack; `quant/matmul.py` has `qbvm_kernel` and `triton_bmm_fA_qB_outer`, but the active PatternKV path calls custom CUDA wrappers. Custom CUDA: `quant/csrc/gemv_cuda.cu` provides packed GEMV and fused-with-base kernels.

12. **Which tensors are saved in KV Cache?**
    `past_key_value` in PatternKV path stores: quantized K transposed, FP16 residual K, K scale, K min, quantized V, FP16 residual V, V scale, V min, total KV length, K assignments, V mask assignments, and V centroid indices.

13. **Pattern index dtype.**
    In Python cache, K assignments and V centroid indices are stored as `torch.long`. Before CUDA K fused call, K assignments are converted to `torch.int16` unless already `uint8/int16/int32`. V indices are converted to `uint8` when centroid count <=256, else `int16`.

14. **V mask dtype.**
    The computed mask is boolean, then stored in cache as `torch.uint8`; the CUDA V fused wrapper also coerces `v_mask_q` to `torch.uint8`.

15. **Any path that only changes values without true compression?**
    Yes, prefill attention output is computed from full FP16 K/V before packing, then the cache is packed for decode. This is expected for prefill. For historical KV after prefill/decode windows, packed `int32` tensors plus scale/min are stored. The legacy `LlamaAttention_PatternKV` class above the flash class implements K/V packing but not PatternKV centroid residuals; active model layers use `LlamaFlashAttention_PatternKV`.

16. **Unused imports, undeclared deps, debug code?**
    `cuml.cluster.KMeans`, `cupy`, `nullcontext`, duplicate `import torch`, `load_dataset` in README/example, and `random` in some quant files appear unused in the real path. `cuml` and `cupy` are not declared in `pyproject.toml`. `quant/test.py` hardcodes `CUDA_VISIBLE_DEVICES="2"`, which is unsuitable for reproducible single-GPU validation. There are many commented debug blocks and old implementations.

17. **Implementation differences vs paper description.**
    Based on code, prefill cluster assignment uses Euclidean distance (`batched_assign`), while decode K/V reassignment uses min-max range distance. V threshold uses the implemented range-contraction inequality. This audit cannot fully verify paper equivalence beyond code-level behavior.

18. **Does the repo provide LongBench, GSM8K, AIME, system performance runners?**
    No. Only `example.py`, quant tests, and timing helper exist. There are no LongBench/GSM8K/AIME/system benchmark runners in this repository.

## Compatibility Risks

- The current server exposes RTX 3090 (SM86), not RTX 4090 (SM89), so requested 4090-specific validation cannot pass on this hardware.
- `patternkv` import will fail because the project metadata name has no corresponding package/module. A minimal wrapper module is needed for the requested import check.
- Top-level `cuml` and `cupy` imports will fail without RAPIDS/CuPy even though the code path uses PyTorch KMeans. They should be made optional or removed if confirmed unused.
- `quant/qmodule.py` imports `dequant_cuda` and `pack`, but these are not provided in this repository; this module is not in the PatternKV attention call path.
