# Centroid Determinism Causal Forensic

## 1. Executive Summary

ROOT_CLASSIFICATION = UNINITIALIZED_OR_STALE_WORKSPACE

The true centroid-builder input and active K centroid are byte-stable; the prior centroid-state nondeterminism is confined to unused/full centroid pool storage that is allocated with `torch.empty` and hashed beyond the active centroid count.

## 2. Current Symptom

Repeated independent B1 prefill previously diverged at layer0 `k_centroid_values`. This run verifies whether the centroid builder input itself changes before assigning root cause.

## 3. Centroid Call Graph

- `models/llama_patternkv.py:1064` `LlamaFlashAttention_PatternKV.forward`: receives layer hidden states, computes Q/K/V projection.
- `models/llama_patternkv.py:1129` reshapes K to `[B, H_kv, seq_len, head_dim]` and applies RoPE.
- `models/llama_patternkv.py:1797` layer prefill compression starts after attention output.
- `models/llama_patternkv.py:1801-1805`: B1 builds `Xmk = key_states.mean(...).permute(...).reshape(n_kv, seq_len, hd).float()`; observed shape is `[8, 384, 128]`.
- `models/llama_patternkv.py:1805` calls `batched_kmeans_fast_compiled(Xmk, k=self.num_k_bases, iters=30, tol=1e-4, seed=0)`.
- `models/llama_patternkv.py:1806-1809` assigns tokens, writes `self.k_base = k_centroids.to(key_states.dtype)`, records `K_CENTROID`.
- `models/llama_patternkv.py:628-721` `batched_kmeans_fast`: local `torch.Generator`, `torch.rand` init, `torch.empty` sums/counts, `zero_`, `scatter_add_`, normalize.

## 4. Input-K Determinism

K_INPUT_DETERMINISTIC = True

| run | sha256 | equal_to_run0 | max_abs_diff | mean_abs_diff | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| 0 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 1 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 2 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 3 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 4 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 5 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 6 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 7 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 8 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 9 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 10 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 11 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 12 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 13 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 14 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 15 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 16 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 17 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 18 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |
| 19 | `cc09f123c45e4481005712c19010b96b6fe57e625ad74e6e1af32bebe4dea30a` | True | 0.0 | 0.0 | 0.0 |

Upstream component unique hashes:

| component | unique_hashes |
| --- | --- |
| `ATTN_INPUT_HIDDEN` | 0 |
| `Q_PROJ` | 1 |
| `K_PROJ` | 1 |
| `V_PROJ` | 1 |
| `K_POST_ROPE` | 1 |
| `KMEANS_K_INPUT` | 1 |
| `K_CENTROID` | 1 |
| `K_ASSIGNMENT` | 1 |
| `CACHE_K_CENTROID_ACTIVE` | 1 |
| `CACHE_K_CENTROID_POOL_FULL_SLOT` | 2 |
| `CACHE_V_CENTROID_ACTIVE` | 1 |
| `CACHE_V_CENTROID_POOL_FULL_SLOT` | 2 |

## 5. Frozen-K Standalone Centroid Test

CENTROID_OPERATOR_100_RUN_UNIQUE_HASHES = 1
ASSIGNMENT_100_RUN_UNIQUE_HASHES = 1

## 6. First Divergent Stage

FIRST_DIVERGENT_STAGE = CACHE_K_CENTROID_POOL_FULL_SLOT

| stage | unique_hashes |
| --- | --- |
| `input_K` | 1 |
| `initial_centroids` | 1 |
| `distance` | 1 |
| `assignment` | 1 |
| `cluster_counts` | 1 |
| `cluster_sums` | 1 |
| `normalized_centroids` | 1 |
| `k_centroid_values` | 1 |

## 7. RNG Oracle

RNG_FIXED: before=1 unique hashes, after=1 unique hashes.

## 8. Workspace Oracle

NORMAL_WORKSPACE_UNIQUE_HASHES = 1
ZEROED_WORKSPACE_UNIQUE_HASHES = 1
FRESH_WORKSPACE_UNIQUE_HASHES = 1

## 9. Synchronization Oracle

NO_SYNC_UNIQUE_HASHES = 1
SYNC_UNIQUE_HASHES = 1

## 10. Atomic / Scatter / Reduction Audit

ATOMIC_REDUCTION_PRESENT = True

- `models/llama_patternkv.py:679` uses `sums.scatter_add_` to accumulate FP32 token values into centroid buckets.
- `models/llama_patternkv.py:686` uses `counts.scatter_add_` for per-cluster counts.
- `models/llama_patternkv.py:655-656` allocates `sums`/`counts` via `torch.empty`, but both are immediately zeroed before accumulation.
- `models/llama_patternkv.py:641-644` uses a local `torch.Generator` with fixed seed for initialization.
- No `tl.atomic_add` appears in the K centroid builder path; CUDA `atomicAdd` occurrences are in fused Value/GEMV kernels, not this prefill K centroid accumulation path.
- PyTorch CUDA `scatter_add_` for floating accumulation is treated as atomic/reduction-like for this forensic classification.

## 11. Deterministic Reference

REFERENCE_UNIQUE_HASHES = 1
PRODUCTION_UNIQUE_HASHES = 1

Production-vs-reference max metrics:

```json
{
  "max_abs_diff_max": 9.5367431640625e-07,
  "mean_abs_diff_max": 1.8144852731438732e-08,
  "rel_l2_max": 4.283145216277262e-08
}
```

## 12. Fixed Reduction Oracle

FIXED_REDUCTION_ORACLE = NOT_APPLICABLE_FOR_CURRENT_PRODUCTION_OPERATOR. The current centroid accumulation path is PyTorch `scatter_add_`; there is no exposed split/chunk topology knob in production code. The deterministic reference provides a fixed-order reduction oracle.

## 13. Root Cause Classification

UNINITIALIZED_OR_STALE_WORKSPACE: `KMEANS_K_INPUT`, `K_CENTROID`, `CACHE_K_CENTROID_ACTIVE`, and frozen-K production centroid are deterministic, but the full centroid pool slot has multiple hashes because inactive capacity is allocated with `torch.empty` and is outside the active centroid count.

## 14. Recommended Production Fix

Option A: deterministic segmented reduction with stable grouping and fixed merge order. Correctness: strongest. Performance: likely slower than scatter in prefill unless optimized. Complexity: medium/high. Batch invariance: strong. Ragged compatibility: good if request-local segments are explicit.

Option B: request-local fixed reduction tree for centroid accumulation. Correctness: strong for serving determinism. Performance: tunable with fixed chunks. Complexity: medium. Batch invariance: strong if partitioning is request/shape invariant. Ragged compatibility: good.

Option C: CPU/simple PyTorch deterministic fallback for debug gates only. Correctness: useful oracle. Performance: poor. Complexity: low. Batch invariance: strong. Ragged compatibility: acceptable only for tests/forensics.

## 15. Next Experiment

MORE FORENSIC: update diagnostic comparators to hash only active centroid counts, then rerun S6-B.3.4D/3.4 to see whether the remaining logit drift has a real semantic state mismatch.

## Environment

```json
{
  "branch": "sys/causal-v4-25-kernel-v1",
  "cuda_available": true,
  "git_head": "cc50fdc513181d2137438cc6a7c0dd8322ccf767",
  "git_status": "?? forensics/\n?? reports/centroid_determinism_causal_forensic.md\n?? scripts/centroid_determinism_causal_forensic.py",
  "gpu": "NVIDIA GeForce RTX 3090",
  "platform": "Linux-7.0.0-28-generic-x86_64-with-glibc2.39",
  "python": "3.10.20",
  "torch": "2.4.1+cu124",
  "torch_cuda": "12.4",
  "triton": "3.0.0"
}
```

```text
Fri Aug 14 19:25:52 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:1A:00.0 Off |                  N/A |
| 30%   27C    P8             20W /  350W |   18409MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        Off |   00000000:1C:00.0 Off |                  N/A |
| 30%   48C    P2            119W /  350W |   17164MiB /  24576MiB |     29%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 3090        Off |   00000000:1D:00.0 Off |                  N/A |
| 30%   31C    P8             19W /  350W |      20MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 3090        Off |   00000000:1E:00.0 Off |                  N/A |
| 30%   31C    P8             22W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA GeForce RTX 3090        Off |   00000000:3E:00.0 Off |                  N/A |
| 30%   31C    P8             23W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA GeForce RTX 3090        Off |   00000000:3F:00.0 Off |                  N/A |
| 30%   28C    P8             28W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA GeForce RTX 3090        Off |   00000000:40:00.0 Off |                  N/A |
| 30%   30C    P8             16W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA GeForce RTX 3090        Off |   00000000:41:00.0 Off |                  N/A |
| 30%   31C    P8             18W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    0   N/A  N/A          201464      C   python                                18380MiB |
|    1   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    1   N/A  N/A         2428872      C   python                                17138MiB |
|    2   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    3   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    4   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    5   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    6   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    7   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
```

## Commands

- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD /data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python scripts/centroid_determinism_causal_forensic.py --device cuda:0`
