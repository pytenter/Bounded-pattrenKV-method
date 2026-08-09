# Causal Attention Design Audit

## Intended Deployable Signal

The requested signal is historical attention received while a token is still in FP16 recent/pending state. This is causal for pseudo-decode because it uses only attention already emitted by the quantized trajectory before the token is packed.

## No-Leakage Rule

The production objective must not use future FP16 attention, full trajectory attention, or Experiment 6 oracle `A_FP`.

## Current Blocker

In the current segmented rolling cache, V centroid assignment is per token x KV-head vector and V affine packing is per token head_dim group. With this granularity:

```text
argmin_c w_i * L(v_i, c) == argmin_c L(v_i, c)
```

for any positive scalar `w_i`. Thus causal attention weighting is mathematically ineffective unless the feasible candidate decision is made over a multi-token tile or another coupled representation choice.

## Static Matched-Path Issue

The existing Experiment 6 static path builds a fresh full prefix cache. It does not expose pack-time historical attention received by each token before packing. Retrofitting that signal would require changing the static execution semantics or adding a new attention-capture production path. This is not a same-objective rescore.

## Gate Decision

`CAUSAL_ATTENTION_NO_LEAKAGE=true` for the pure helper tests, but `STATIC_IMPORTANCE_MATCHED_PATH_VALID=false` and `V_CAUSAL_ATTN_EFFECTIVE_UNDER_CURRENT_GRANULARITY=false`. Formal 8-GPU screening is therefore not approved.
