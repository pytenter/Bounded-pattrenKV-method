# Tensor Shape Trace

PatternKV-paper B2 prefill:

| Boundary | Tensor | Shape | Owner |
|---|---|---:|---|
| K/V projection | `value_states` | `[2,8,512,128]` | active batch row |
| V k-means | `self.v_centroids` | `[2,8,32,128]` initially | request-local bank |
| packed history | `v_assignment_idx` | `[2,8,384]` | request-local assignment |
| packed history | `v_pattern_mask` | `[2,8,384]` | request-local gate |
| dynamic/page bank | `v_centroids` | `[2,8,48,128]` in failing path | request-local bank plus dynamic capacity |
| old reader | `cuda_attn_v_fused_with_base` | expected `[8,C,128]` | shared bank assumption |
| fixed reader path | `operator_ready_page_pools.centroids` | `[2,8,48,128]` | request-local bank |

The first inconsistent boundary was the non-mixed Value attention reader dispatch, not cache construction.
