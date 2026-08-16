# V Causal Importance Update Contract

`mass = attn_weights.detach().float().mean(dim=1).sum(dim=1)`. The current implementation pads/casts `cache.v_causal_importance` to `cache.total_tokens`, then applies `cache.v_causal_importance[:, :width] += mass[:, :width]`. For ragged segmented batches this treats the physical concatenated attention axis as if it were already each request's logical token axis.
