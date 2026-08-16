# Capture Feasibility Audit

Mutable decode objects:

- Input token IDs: STATIC_ADDRESS_MUTABLE_VALUE.
- Position IDs/query positions: PYTHON_CONTROL_FLOW in model wrapper, values derived from cache length.
- Request-local positions and active batch row mapping: STATIC for B1, PYTHON_CONTROL_FLOW for dynamic serving.
- Slot IDs/request IDs: STATIC for B1 graph, PYTHON_CONTROL_FLOW for lifecycle changes.
- KV valid lengths/sink/pending/recent/historical counts: HOST_SCALAR plus STATIC_ADDRESS_MUTABLE_VALUE inside serialized cache.
- Cache write indices and page metadata: DYNAMIC_ADDRESS in eager cache mutation because `torch.cat` creates replacement tensors.
- Centroid counts/valid lengths: CUDA_SIDE_STATE plus HOST_SCALAR in update paths.
- Attention masks/score shapes/workspaces/output buffers: DYNAMIC_SHAPE across decode steps in eager, graph sequence captures one fixed step shape per replay position.

GRAPH_BLOCKING_HOST_SCALARS found during probing:

- `request_invariant_segmented_attention_softmax`: `totals.max().item()` blocked full capture before the fixed-split CUDA softmax path. It was moved out of the fixed-split path.
- `request_invariant_full_value_attention`: `lengths.max().item()` blocked Value tail capture. It was replaced with the physical segment width, which is semantically equivalent because invalid lanes are masked.
