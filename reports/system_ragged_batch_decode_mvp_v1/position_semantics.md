# Position Semantics

Per-request decode positions are computed by the metadata helper, but production `LlamaModel_PatternKV` still derives default positions from one global past length.
