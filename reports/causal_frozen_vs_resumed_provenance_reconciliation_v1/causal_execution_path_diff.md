# CAUSAL Execution Path Diff

| Stage | 8d path | 50a path | Same/different |
|---|---|---|---|
| Loader | `load_causal_model` | `load_causal_model` | Same |
| Method adapter | `PatternKVAdapter` | `PatternKVAdapter` | Same for CAUSAL |
| Prefill | active-batch prefill when enabled | active-batch prefill when enabled | Same |
| Cache path/mode | segmented rolling | segmented rolling | Same |
| Decode timing | prefill before timed window | prefill before timed window | Same |
| QK reader | CAUSAL compressed-domain reader | CAUSAL compressed-domain reader | Same by direct A/B |
| Softmax | fixed split enabled | fixed split enabled | Same |
| Value | mixed fused page Value | mixed fused page Value | Same |
| FP16 tail | tail fusion enabled | tail fusion enabled in reproduced A/B | Same |
| Cache append/update | segmented cache append | segmented cache append | Same by counters and A/B |
| Page pool | operator-ready mixed page pool | operator-ready mixed page pool | Same for CAUSAL |
| Allocator lifecycle | `expandable_segments:True` set by frozen runner | not set by current paper wrapper unless inherited | Different |
