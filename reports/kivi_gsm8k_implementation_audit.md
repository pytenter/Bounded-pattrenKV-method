# KIVI GSM8K Implementation Audit

- created_at: `2026-08-03T13:43:49Z`
- repo: `/data/zypan/PatternKV-repro`
- current_result_dir: `results/gsm8k/smoke_1024`
- official_kivi_repo_checked: `https://github.com/jy-yuan/KIVI`
- official_paper_checked: `KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache`

## Current GSM8K Smoke Result

| method | rows | correct | accuracy | delta_vs_fp16 | truncated | normal_eos_rate | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 50 | 41 | 82.0% | 0.0 pts | 1 | 98.0% | 0 |
| kivi | 50 | 18 | 36.0% | -46.0 pts | 20 | 60.0% | 0 |
| patternkv | 50 | 42 | 84.0% | +2.0 pts | 1 | 98.0% | 0 |

The KIVI rows are complete and have no parser failures or runtime errors. The main failure mode is generation degeneration: 20/50 KIVI samples hit the 1024-token limit, often after entering repeated reasoning loops.

## Official KIVI Paper Numbers

The paper's LM-Eval table reports CoQA, TruthfulQA, and GSM8K. For GSM8K, official KIVI does not show a collapse comparable to the current local run:

| model | 16bit GSM8K | KIVI setting | KIVI GSM8K |
|---|---:|---|---:|
| Llama-2-7B | 13.50 | KIVI-2 R128 | 12.74 |
| Llama-2-7B | 13.50 | KIVI-2 R32 | 13.57 |
| Llama-2-13B | 22.67 | KIVI-2 R128 | 20.77 |
| Llama-2-13B | 22.67 | KIVI-2 R32 | 20.62 |
| Falcon-7B | 4.55 | KIVI-4 R128 | 4.47 |
| Falcon-7B | 4.55 | KIVI-4 R32 | 3.94 |
| Mistral-7B | 38.36 | KIVI-2 R128 | 36.01 |
| Mistral-7B | 38.36 | KIVI-2 R32 | 34.34 |

The paper also states that KIVI keeps a full-precision sliding window, and that this sliding window is important for hard tasks such as GSM8K.

## Official Code Path

The official repository uses model-specific KIVI classes:

- `models/llama_kivi.py::LlamaForCausalLM_KIVI`
- `models/mistral_kivi.py::MistralForCausalLM_KIVI`
- `models/falcon_kivi.py::FalconForCausalLM_KIVI`

The official Llama-3.1-8B GSM8K example uses:

- model: `LlamaForCausalLM_KIVI`
- prompt: 5-shot GSM8K
- `k_bits=2`
- `v_bits=2`
- `group_size=32`
- `residual_length=32`
- `max_new_tokens=96`
- `use_flash=True`

The official LongBench note recommends KIVI-2 for vanilla MHA models, but KIVI-4 for MQA/GQA models because their KV cache is already compressed. Llama-3.1-8B-Instruct is a GQA model, so 2-bit KV is an aggressive setting for quality.

## Current Local Code Path

The local GSM8K runner does not currently use the official KIVI model class. It uses standard HuggingFace `LlamaForCausalLM` plus `FlexibleVanillaQuantizedCache` from kvtuner:

- local file: `bench/bench_gsm8k_patternkv.py`
- model load: standard `LlamaForCausalLM`
- cache: `FlexibleVanillaQuantizedCache`
- current config: `k_bits=2`, `v_bits=2`, `group_size=128`, `residual_length=128`, `axis_key=1`, `axis_value=0`, `asym=True`
- generation passes the cache through `past_key_values=args.cache_factory()`

This implementation should be treated as a KIVI-like flexible quantized-cache baseline, not as a faithful official KIVI implementation.

## Key Mismatches

1. Official KIVI modifies attention itself and uses packed quantized cache GEMV kernels. The local path dequantizes quantized history back into tensors and feeds HuggingFace attention.

2. Official KIVI appends newly quantized residual chunks/tokens to the packed quantized cache. The local flexible cache dequantizes the old quantized history, concatenates it with the residual window and the new token, then requantizes the whole returned history when the residual window fills. This can accumulate quantization error during long generation.

3. Official KIVI preserves a recent full-precision KV sliding window after prefill and during decode. In the local non-`force_quant` path, the first update quantizes the whole prefill cache and stores empty fp16 key/value caches, so the first decode steps do not exactly match the official residual-cache behavior.

4. The local cache implementation's own docstring says its quantization is unlike the KIVI paper for the generic class: it describes per-channel grouping for both keys and values, in contrast to the paper. The local axes were set to a more KIVI-like key/value orientation, but this still is not the official algorithm or kernel path.

5. The current experiment prompt/protocol is not the official GSM8K protocol. We use the local zero-shot CoT prompt and `max_new_tokens=1024`; official example uses 5-shot GSM8K and `max_new_tokens=96`.

6. The current run uses KIVI-2 on Llama-3.1-8B-Instruct, a GQA model. The official LongBench guidance says KIVI-4 is preferred for MQA/GQA models when preserving full-precision performance is the goal.

## Diagnosis

The current KIVI GSM8K result is very likely an implementation/protocol mismatch rather than evidence that official KIVI performs poorly. The strongest evidence is:

- FP16 and PatternKV stop normally on 49/50 samples, but KIVI stops normally on only 30/50.
- KIVI has 20 length truncations and repeated reasoning loops.
- Official paper numbers show small drops on GSM8K, not a 46-point drop versus FP16.
- Local code path is not the official KIVI attention/model implementation.

## Recommended Fix

To make the KIVI baseline credible, replace or add the local KIVI path with an official-KIVI path:

1. Vendor or import the official KIVI model code and CUDA quant kernels into the repo.
2. Load KIVI through `LlamaForCausalLM_KIVI`, not standard `LlamaForCausalLM` plus `FlexibleVanillaQuantizedCache`.
3. Add explicit configs:
   - `kivi2_g32_r32`: `k_bits=2`, `v_bits=2`, `group_size=32`, `residual_length=32`
   - `kivi2_g32_r128`: paper long-context setting
   - `kivi4_g32_r128`: recommended quality setting for GQA/MQA-style KV compression
4. For paper comparison, add a 5-shot GSM8K protocol matching the official example.
5. Keep the current flexible-cache baseline, but rename it to avoid claiming it is official KIVI.
6. Add a small parity test: with quantization disabled or with an all-fp16 residual-only path, KIVI logits for a short prompt should match FP16 within tolerance.

