# Branch Map

## Core Branches

| Branch | Purpose | Policy |
|---|---|---|
| `main` | Public/reproduction project entry. | Not automatically promoted by repository hygiene tasks. Keep reproduction-focused until a curated release is selected. |
| `exp/causal-v4-25-generalization-v1` | Frozen quality/generalization evidence for CAUSAL-V4@25%. | Do not mix system optimization back into it. Expected SHA: `3dcedb42674140296c47cd56cf6ccbc1017474bc`. |
| `sys/causal-v4-25-kernel-v1` | Active systems/runtime development branch. | Continue runtime/serving/forensic work here; push only to `bounded`. |
| `release/causal-v4-25-system-v1` | Stable system checkpoint before segmented heterogeneous attention state-merge optimization. | Immutable checkpoint for the validated pre-state-merge system. |

## Historical Branches

| Branch | Purpose | Status | Unique commits? | Recommended action |
|---|---|---|---|---|
| `main` | Public/reproduction project entry. | CORE | no | KEEP |
| `sys/causal-v4-25-kernel-v1` | Active systems/runtime development branch. | CORE | no | KEEP |
| `exp/aime24-full-causal25-quality-4gpu` | Historical experiment branch. | FROZEN_SCIENTIFIC | no | FREEZE |
| `exp/causal-v4-25-generalization-v1` | Frozen CAUSAL-V4@25% quality/generalization evidence. | FROZEN_SCIENTIFIC | no | FREEZE |
| `analysis/patternkv-vgate-layer-head-opportunity` | Historical analysis branch. | HISTORICAL_EXPERIMENT | yes | REVIEW_REQUIRED |
| `exp/aime-int2-wave1-v100-8gpu` | Historical experiment branch. | HISTORICAL_EXPERIMENT | yes | REVIEW_REQUIRED |
| `exp/aime-pattern-hadamard-mechanism-3090` | Historical experiment branch. | HISTORICAL_EXPERIMENT | yes | REVIEW_REQUIRED |
| `exp/aime-pattern-varn-mechanism-3090` | Historical experiment branch. | HISTORICAL_EXPERIMENT | yes | REVIEW_REQUIRED |
| `exp/aime-pseudodecode-3090-8gpu` | Historical experiment branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `exp/aime-qk-routing-vdirection-3090` | Historical experiment branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `exp/aime-selective-value-precision-3090` | Historical experiment branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `exp/aime-value-capacity-budget-3090` | Historical experiment branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `exp/aime-value-direction-screen-3090` | Historical experiment branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `exp/aime-value-objective-screen-3090` | Historical experiment branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `exp/patternkv-4090-range-aware-targeted` | Historical experiment branch. | HISTORICAL_EXPERIMENT | yes | REVIEW_REQUIRED |
| `exp/patternkv-insight-wave-a-4090-runtime6c88` | Historical experiment branch. | HISTORICAL_EXPERIMENT | yes | REVIEW_REQUIRED |
| `exp/patternkv-insight-wave-a-4gpu` | Historical experiment branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `exp/patternkv-longbench-data-parity-wave-a` | Historical experiment branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `exp/patternkv-parity-microsmoke-wave-a` | Historical experiment branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `insight/patternkv-diagnostics-v1` | Historical insight/diagnostic development branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `insight/patternkv-observer-wave-a` | Historical insight/diagnostic development branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `insight/patternkv-runner-parity-wave-a` | Historical insight/diagnostic development branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `repro/patternkv-longbench-8k-single4090` | Historical reproduction/evaluation branch. | HISTORICAL_EXPERIMENT | no | ARCHIVE_CANDIDATE |
| `release/causal-v4-25-system-v1` | Stable pre-state-merge system checkpoint. | RELEASE | no | KEEP |
