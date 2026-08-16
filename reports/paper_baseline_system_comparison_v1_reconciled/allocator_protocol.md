# Allocator Protocol

Every worker subprocess is launched with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `allocator_protocol_valid` is true only when the worker environment explicitly contains that value; invalid allocator rows fail formal validation.
