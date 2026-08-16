# Production Fix

`models/segmented_cache.py` adds `request_invariant_segmented_attention_softmax`; `models/llama_patternkv.py` calls it at the segmented decode attention softmax site. The helper scatters physical segment logits into request-local logical order, performs fixed-size online softmax state merge by logical split index, then scatters probabilities back to physical layout.
