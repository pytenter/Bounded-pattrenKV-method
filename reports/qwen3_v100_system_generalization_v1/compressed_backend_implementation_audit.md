# Compressed Backend Implementation Audit

Added `models/qwen3_patternkv_system.py` with native Qwen3 compressed attention classes, counters, and no calls to `reconstruct_full_k` / `reconstruct_full_v` in the compressed decode path. The original reference adapter remains unchanged. Runtime import uses the server-side Transformers 4.51 vendor path when the target worktree has no local vendor runtime.
