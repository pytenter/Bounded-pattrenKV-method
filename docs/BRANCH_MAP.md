# Branch Map

## Frozen Branch Model

| Branch | Role | Policy |
|---|---|---|
| `main` | Public/reproduction entry. | Untouched by this freeze task. |
| `exp/causal-v4-25-generalization-v1` | Frozen quality/generalization scientific evidence. | Keep frozen at its recorded SHA; no system-runtime edits. |
| `sys/causal-v4-25-kernel-v1` | Frozen system research history and paper-assembly source. | No new throughput engineering; push only to `bounded`. |
| `release/causal-v4-25-system-v1` | Immutable pre-state-merge system checkpoint. | Preserve as the validated system checkpoint before the final serving freeze. |
| `release/causal-v4-25-system-final` | Immutable final full-model serving experiment checkpoint. | Keep pinned to `8d60485b5d2c93b7c1d478efc449de56d28159c3`. |

## Retained Scientific History

| Branch | Status | Action |
|---|---|---|
| `analysis/patternkv-vgate-layer-head-opportunity` | Unique scientific provenance | KEEP |
| `exp/aime-int2-wave1-v100-8gpu` | Frozen scientific history | KEEP |
| `exp/aime-pattern-hadamard-mechanism-3090` | Frozen scientific history | KEEP |
| `exp/aime-pattern-varn-mechanism-3090` | Frozen scientific history | KEEP |
| `exp/aime-pseudodecode-3090-8gpu` | Historical experiment history | KEEP |
| `exp/aime-qk-routing-vdirection-3090` | Historical experiment history | KEEP |
| `exp/aime-selective-value-precision-3090` | Historical experiment history | KEEP |
| `exp/aime-value-capacity-budget-3090` | Historical experiment history | KEEP |
| `exp/aime-value-direction-screen-3090` | Historical experiment history | KEEP |
| `exp/aime-value-objective-screen-3090` | Historical experiment history | KEEP |
| `exp/aime24-full-causal25-quality-4gpu` | Frozen scientific history | KEEP |
| `exp/patternkv-4090-range-aware-targeted` | Unique scientific provenance | KEEP |
| `exp/patternkv-insight-wave-a-4090-runtime6c88` | Unique scientific provenance | KEEP |
| `exp/patternkv-insight-wave-a-4gpu` | Historical experiment history | KEEP |
| `exp/patternkv-longbench-data-parity-wave-a` | Historical experiment history | KEEP |
| `exp/patternkv-parity-microsmoke-wave-a` | Historical experiment history | KEEP |
| `insight/patternkv-diagnostics-v1` | Historical diagnostic history | KEEP |
| `insight/patternkv-observer-wave-a` | Historical diagnostic history | KEEP |
| `insight/patternkv-runner-parity-wave-a` | Historical diagnostic history | KEEP |
| `repro/patternkv-longbench-8k-single4090` | Historical reproduction history | KEEP |

## Final Branch Anchors

| Branch | SHA | Meaning |
|---|---|---|
| `sys/causal-v4-25-kernel-v1` | `8d60485b5d2c93b7c1d478efc449de56d28159c3` | Frozen final system history. |
| `release/causal-v4-25-system-final` | `8d60485b5d2c93b7c1d478efc449de56d28159c3` | Exact final full-model serving checkpoint. |
| `release/causal-v4-25-system-v1` | `0ca6debff700f68ae8ff536e77ddb2cb1e68d69d` | Pre-state-merge system checkpoint. |
| `exp/causal-v4-25-generalization-v1` | `3dcedb42674140296c47cd56cf6ccbc1017474bc` | Frozen quality/generalization checkpoint. |
