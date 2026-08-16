# KV Runtime vs Full-Model Path Diff

| Question | Isolated KV runtime | Full-model CAUSAL path | Evidence |
| --- | --- | --- | --- |
| Metadata/cache rebuild frequency | Synthetic pool allocation once per run/slot model | Rebuilds per layer per decode token during `assemble_batch` and again during `split_batch` | `_merge_operator_ready_page_pools`: 32 calls/token; `_slice_operator_ready_page_pools_for_request`: 32 calls/token |
| Row/cache copies | Does not round-trip full model per-layer cache wrappers | Copies request cache tensors on split and concatenates/pads tensors on assemble | 447.47 MB copied/allocated per generated token across assembly/split wrappers |
| Fused Value operator | Runtime models compressed KV capacity directly | Fused page operator is used but surrounded by full-model softmax/cache wrapper work | `fused_page_operator_calls=128`; wrapper timing only 6.28 ms/token |
| Full transformer work | Not included | Embedding, 32 layers, MLP, norms, LM head included | Path audit and layer timing |
| Centroid state lifetime | Not modeled as full per-layer request pool | Retains per-layer `PatternKVCentroidStatePool` with default 16 slots and 512 dynamic centroids | 1.212 GB retained centroid state at context2048/B1 |
