# Runtime Dataflow

Q/K/V projection feeds selector and cache append. Packed K remains in the tight INT2 path. QK and softmax produce attention weights. The `fused_page` backend consumes those weights plus operator-ready V2/V4 page pools and returns the attention Value output. Post-attention hidden and logits are compared in `correctness_runs.csv`.
