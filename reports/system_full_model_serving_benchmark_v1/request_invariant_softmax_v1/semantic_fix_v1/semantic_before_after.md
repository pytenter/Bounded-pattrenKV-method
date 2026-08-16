# Semantic Before/After

Before fix: opt-in fixed-split CUDA path had B1 and B2 normal pass but failed multi-request ragged membership/order gates. First production divergence for the remaining B4 case was request C, step 14, layer 0, `ATTENTION_PRE_O_PROJ`; softmax input scores already differed.

Forensic narrowing showed C's pre-flush pending K and valid centroid entries were request-local exact, but newly packed K diverged after a prior peer flush. Root cause: single-slot row cache centroid views used batch-level scalar update counts instead of slot-local `centroid_state_pool.k_counts/v_counts`, exposing stale centroid tail entries during assignment.

After fix: ragged gate classification is `PATTERNKV_RAGGED_MULTI_STEP_CORRECTNESS_SUPPORTED` with B2 reorder `True`, B4 `True`, and flush `True`.
